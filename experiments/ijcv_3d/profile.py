"""Measure one IJCV 3D checkpoint under a recorded GPU protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

os.environ['HDMAPNET_TASK_FAMILY'] = 'lane'
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.flop_counter import FlopCounterMode

from data.datasets.lane.openlane.openlane import OpenLaneDataset
from experiments.ijcv_3d.data import openlane_transforms_3d, validate_geometry
from experiments.ijcv_3d.model import build_model
from experiments.ijcv_3d.io import autocast_dtype, autocast_precision, move, write_json


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-version', default='3d_revision_v1')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Existing measurement evidence is never overwritten')
    if args.warmup < 1 or args.iterations < 5:
        raise ValueError('Use at least one warm-up and five timed iterations')
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    validate_geometry(config)
    device = torch.device('cuda:0')
    transform, collate = openlane_transforms_3d(train=False)
    dataset = OpenLaneDataset(str(args.data_root), 'val', transform, version=args.cache_version)
    loader = DataLoader(Subset(dataset, [0]), batch_size=1, num_workers=0, collate_fn=collate)
    batch = move(next(iter(loader)), device)
    model = build_model(config).to(device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state.get('config') != config:
        raise ValueError('Checkpoint and requested configuration differ')
    saved_model = {key.removeprefix('module.'): value for key, value in state['model'].items()}
    model.load_state_dict(saved_model)
    model.eval()
    git = lambda *command: subprocess.check_output(['git', *command], text=True).strip()
    protocol = dict(
        kind='single_gpu_checkpoint_complexity', config=config, checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha256(args.checkpoint), checkpoint_epoch=state.get('epoch'),
        code_commit=git('rev-parse', 'HEAD'), dirty=git('status', '--porcelain'),
        sample=dataset.data_infos[0]['img_path'], batch_size=1, precision=autocast_precision(),
        input_shape=list(batch['img'].shape), bev_feature_shape=config.get('bev_feature_shape'),
        warmup=args.warmup, iterations=args.iterations, preprocessing_and_file_io_included=False,
        model_postprocessing_included=True, cuda_synchronized_each_iteration=True,
        cpu_numpy_postprocessing_excluded_from_flop_count=True,
        gpu=torch.cuda.get_device_name(0), torch=torch.__version__, python=platform.python_version(),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        state_bytes=sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()),
        note='FLOPs are operations attributed by torch.utils.flop_counter; unsupported operations are omitted.')
    write_json(args.output / 'protocol.json', protocol)
    with torch.no_grad():
        for _ in range(args.warmup):
            with torch.autocast('cuda', dtype=autocast_dtype()):
                model(batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        elapsed_ms = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            with torch.autocast('cuda', dtype=autocast_dtype()):
                model(batch)
            torch.cuda.synchronize()
            elapsed_ms.append((time.perf_counter_ns() - started) / 1e6)
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.synchronize()
    # PyTorch 2.8's ModuleTracker requires an autograd graph when counting
    # module FLOPs. The postprocessors convert tensors to
    # NumPy, so bypass only that CPU portion for this single profiling pass.
    # The timed iterations above still include the real postprocessor.
    if hasattr(model, 'detector'):
        original_postprocess = model.detector.postprocess
        model.detector.postprocess = lambda *unused_args, **unused_kwargs: ([], [], [])
        restore_target = model.detector
    else:
        original_postprocess = model.postprocess
        model.postprocess = lambda unused_outputs: dict(
            lane_points=[], lane_attris=[], lane_scores=[])
        restore_target = model
    try:
        with FlopCounterMode(display=False) as counter:
            with torch.autocast('cuda', dtype=autocast_dtype()):
                flop_output = model(batch)
        torch.cuda.synchronize()
        del flop_output
    finally:
        restore_target.postprocess = original_postprocess
    result = dict(
        latency_ms=dict(mean=statistics.fmean(elapsed_ms), median=statistics.median(elapsed_ms),
                        p90=percentile(elapsed_ms, 90), minimum=min(elapsed_ms), maximum=max(elapsed_ms)),
        throughput_fps=1000 / statistics.fmean(elapsed_ms), peak_allocated_bytes=peak,
        peak_allocated_gib=peak / 2**30, attributed_flops=int(counter.get_total_flops()),
        attributed_gflops=counter.get_total_flops() / 1e9, raw_latency_ms=elapsed_ms)
    write_json(args.output / 'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'raw_latency_ms'}), flush=True)


if __name__ == '__main__':
    main()
