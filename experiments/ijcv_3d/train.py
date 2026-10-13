"""Distributed training and complete OpenLane validation export."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
os.environ['HDMAPNET_TASK_FAMILY'] = 'lane'
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from data.datasets.lane.openlane.openlane import OpenLaneDataset
from experiments.ijcv_3d.data import openlane_transforms_3d, validate_geometry
from experiments.ijcv_3d.model import build_model
from experiments.ijcv_3d.geometry import bev_points_to_ego
from experiments.ijcv_3d.io import autocast_dtype, autocast_precision, move, write_json
from tools.metrics.lane.openlane.lane3d.eval_3D_lane import LaneEval
from types import SimpleNamespace


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(dataset, root):
    for info in dataset.data_infos:
        image = root / info['img_path']
        annotation = root / 'lane3d_1000' / Path(info['img_path'].removeprefix('images/')).with_suffix('.json')
        if not image.is_file() or image.stat().st_size == 0 or not annotation.is_file():
            raise FileNotFoundError(f'Incomplete data: {image} / {annotation}')


def evaluate(model, dataset, collate, root, output, rank, world, device):
    model.eval()
    subset = Subset(dataset, list(range(rank, len(dataset), world)))
    loader = DataLoader(subset, batch_size=1, num_workers=2, collate_fn=collate)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            meta = batch['img_metas'][0]
            with torch.autocast('cuda', dtype=autocast_dtype()):
                pred = model(move(batch, device))
            name = 'validation/' + meta['filename'].replace(',', '/')
            lanes = []
            for lane, attr, score in zip(pred['lane_points'][0], pred['lane_attris'][0], pred['lane_scores'][0]):
                if pred.get('lane_points_frame') == 'ego':
                    xyz = np.asarray(lane)
                else:
                    points, _ = lane
                    # Inverse of the exact transform used to rasterize BEV targets.
                    xyz = bev_points_to_ego(points, meta['calibration']['trans_bev2ego'])
                lanes.append(dict(xyz=xyz.tolist(), category=int(attr), score=float(score)))
            path = output / Path(name).with_suffix('.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(file_path=name, lane_lines=lanes)))
    dist.barrier()
    stats = None
    if rank == 0:
        names = [i['img_path'].removeprefix('images/') for i in dataset.data_infos]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate validation identities')
        for name in names:
            if not (output / Path(name).with_suffix('.json')).is_file():
                raise FileNotFoundError(f'Missing validation prediction: {name}')
        listing = output / 'test_list.txt'
        listing.write_text('\n'.join(names) + '\n')
        evaluator = LaneEval(SimpleNamespace(dataset_dir=str(root / 'lane3d_1000'),
                                            pred_dir=str(output), test_list=str(listing)))
        values = evaluator.bench_one_submit(str(output) + '/', str(root / 'lane3d_1000') + '/', str(listing))
        keys = ['F1', 'recall', 'precision', 'category_accuracy', 'x_error_close', 'x_error_far', 'z_error_close', 'z_error_far',
                'recall_matches', 'precision_matches', 'category_matches', 'gt_lanes', 'predicted_lanes', 'matches']
        stats = dict(zip(keys, [float(v) if np.isfinite(v) else None for v in values]))
        stats['validation_samples'] = len(dataset)
        write_json(output / 'metrics.json', stats)
        print(json.dumps(dict(event='validation', **stats)), flush=True)
    dist.barrier()
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs' / 'complete.json')
    p.add_argument('--cache-version', default='3d')
    p.add_argument('--epochs', type=int)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--allow-pilot-subset', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--init-common-from', type=Path,
                   help='Baseline checkpoint whose O-producing modules initialize a direct-3D control')
    p.add_argument('--freeze-common', action='store_true',
                   help='Freeze the initialized O-producing modules for a same-O decoder diagnostic')
    args = p.parse_args()
    rank, local, world = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=2), device_id=device)
    torch.set_num_threads(4)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cudnn.benchmark = False
    config = json.loads(args.config.read_text())
    validate_geometry(config)
    epochs = args.epochs or config['epochs']
    if config['batch_size'] % world:
        raise ValueError('Global batch size must be divisible by world size')
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=args.resume is not None)
    dist.barrier()
    train_tf, collate = openlane_transforms_3d(train=True)
    val_tf, val_collate = openlane_transforms_3d(train=False)
    # Never generate a misleading full-data cache while a transfer is incomplete.
    for split in ('train', 'val'):
        if not (args.data_root / f'openlane_infos_{split}_{args.cache_version}.pkl').is_file():
            raise FileNotFoundError(f'Missing prepared {split} cache')
    train = OpenLaneDataset(str(args.data_root), 'train', train_tf, version=args.cache_version)
    val = OpenLaneDataset(str(args.data_root), 'val', val_tf, version=args.cache_version)
    if not args.allow_pilot_subset and (len(train) < 100000 or len(val) < 10000):
        raise ValueError('Refusing a benchmark run on a small subset')
    if rank == 0:
        verify_dataset(train, args.data_root)
        verify_dataset(val, args.data_root)
    dist.barrier()
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, seed=args.seed, shuffle=True)
    loader = DataLoader(train, batch_size=config['batch_size'] // world, sampler=sampler,
                        num_workers=args.workers, collate_fn=collate, drop_last=True)
    model = build_model(config).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    initialized_common = None
    if args.init_common_from:
        if args.resume:
            raise ValueError('--resume and --init-common-from are mutually exclusive')
        if not hasattr(model, 'load_common_state'):
            raise ValueError('--init-common-from requires a model with load_common_state')
        baseline = torch.load(args.init_common_from, map_location='cpu', weights_only=False)
        initialized_common = model.load_common_state(baseline.get('model', baseline))
    if args.freeze_common:
        if not hasattr(model, 'freeze_common'):
            raise ValueError('--freeze-common requires a model with freeze_common')
        model.freeze_common()
    optimizer = torch.optim.AdamW(model.param_groups(config['lr'], config['weight_decay']), betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [config['lr_drop_epoch']], gamma=.1)
    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['config'] != config or state['world_size'] != world:
            raise ValueError('Resume config/world size mismatch')
        if state.get('common_representation_frozen', False) != args.freeze_common:
            raise ValueError('Resume common-representation freeze mode mismatch')
        saved_model = {key.removeprefix('module.'): value for key, value in state['model'].items()}
        model.load_state_dict(saved_model)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        rng = state['rng'][rank]
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'], device)
        start_epoch = state['epoch']
    # Support the later image-supervision ablation, where a prediction branch
    # remains present but receives no training gradient.
    ddp = DistributedDataParallel(model, device_ids=[local], find_unused_parameters=True)
    git = lambda *cmd: subprocess.check_output(['git', *cmd], text=True).strip()
    if rank == 0:
        manifest = dict(kind='subset_pipeline_check' if args.allow_pilot_subset else 'ijcv_3d_training',
                        args=vars(args), config=config, code_commit=git('rev-parse', 'HEAD'),
                        dirty=git('status', '--porcelain'), train_samples=len(train), validation_samples=len(val),
                        seed=args.seed, world_size=world, actual_epochs=epochs,
                        initialized_common_keys=initialized_common,
                        common_representation_frozen=args.freeze_common,
                        precision=f'{autocast_precision()}; FP32 losses and voting', gradient_clip_norm=35.,
                        batch_normalization='SyncBatchNorm', parameters=sum(p.numel() for p in model.parameters()),
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        data_sha256={s: sha256(args.data_root / f'openlane_infos_{s}_{args.cache_version}.pkl') for s in ('train', 'val')})
        write_json(args.output / f'manifest_epoch_{start_epoch:03d}.json', manifest)
        (args.output / 'working_tree.patch').write_text(git('diff', 'HEAD'))
        with (args.output / 'environment.txt').open('w') as f:
            subprocess.run([os.sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=True)
    dist.barrier()
    log = (args.output / f'train_rank{rank}.jsonl').open('a', buffering=1)
    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        ddp.train()
        for step, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=autocast_dtype()):
                losses = ddp(move(batch, device))
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss epoch={epoch} step={step} rank={rank}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 35., error_if_nonfinite=True)
            optimizer.step()
            if step % 20 == 0 or step == len(loader) - 1:
                row = dict(epoch=epoch + 1, step=step + 1, steps_per_epoch=len(loader),
                           loss=float(loss.detach()), losses={k: float(v.detach()) for k, v in losses.items()},
                           grad_norm=float(norm), elapsed_seconds=time.monotonic() - started,
                           peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
                log.write(json.dumps(row) + '\n')
                if rank == 0:
                    print(json.dumps(row), flush=True)
        scheduler.step()
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == epochs:
            evaluate(model, val, val_collate, args.data_root, args.output / f'validation_{epoch+1:03d}', rank, world, device)
        # Capture RNG after validation as it also constructs data-loader iterators.
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state(device))
        states = [None] * world
        dist.all_gather_object(states, rng)
        if rank == 0:
            checkpoint = args.output / f'epoch_{epoch+1:03d}.pt'
            tmp = checkpoint.with_suffix('.tmp')
            torch.save(dict(model=model.module.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            epoch=epoch + 1, config=config, world_size=world, rng=states,
                            common_representation_frozen=args.freeze_common), tmp)
            tmp.replace(checkpoint)
        dist.barrier()
    if rank == 0:
        write_json(args.output / 'summary.json', dict(status='completed', epochs=epochs,
                   kind='subset_pipeline_check' if args.allow_pilot_subset else 'ijcv_3d_training',
                   elapsed_seconds=time.monotonic() - started))
    log.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
