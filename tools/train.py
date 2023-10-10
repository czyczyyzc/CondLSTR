import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import random
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.backends.cuda as cuda
import torch.backends.cudnn as cudnn
import torch.utils.tensorboard as tensorboard
import numpy as np
from contextlib import suppress
from modeling import models
from data import datasets, transforms, dataloaders
from engine.trainer import Trainer
from engine.tester import Tester
from engine.evaluator import Evaluator
from utils.logger import Logger
from utils.serialization import load_checkpoint, save_checkpoint


def argument_parser():
    parser = argparse.ArgumentParser(description='HDMapNet with Pytorch Implementation')
    parser.add_argument('--gpu-ids', type=str, default='0')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='laion', choices=datasets.names())
    parser.add_argument('-v', '--version', type=str, default='v1.0')
    parser.add_argument('-j', '--num-workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', nargs='+', type=int, default=[4])
    parser.add_argument('-td', '--test-dataset', type=str, default=None, choices=datasets.names())
    parser.add_argument('-tb', '--test-batch-size', type=int, default=None)
    parser.add_argument('-pf', '--prefetch-factor', type=int, default=2, help="batches prefetched by each worker")
    # model
    parser.add_argument('-a', '--arch', type=str, default='SamClipMix', choices=models.names())
    parser.add_argument('-t', '--task', type=str, default='segmentation', help="the task to carry out")
    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['amp', 'amp_bf16', 'fp32'],
                        help="floating point precision")
    parser.add_argument('-c', '--num-classes', type=int, default=1, help="number of classes")
    # optimizer
    parser.add_argument('-e', '--num-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4, help="initial learning rate")                  # 5e-4
    parser.add_argument('--optim', type=str, default='AdamW', help="optimizer")
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999], help="betas in adamw")   # (0.9, 0.98)
    parser.add_argument('--eps', type=float, default=1e-8, help="numerical stability term in adamw")     # 1e-6
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4)                               # 0.2 # 0.001
    parser.add_argument('--nesterov', action='store_true', default=False)
    parser.add_argument('--max-grad', type=float, default=0.1, help="maximum gradient")                  # 10  # 35
    parser.add_argument('--schedule', type=str, default='step', help="learning schedule")                # cosine
    parser.add_argument('--lr-min', type=float, default=1e-5)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=10000, help="number of steps to warmup")
    # training
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--eval-epoch', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
    parser.add_argument('--load-model-only', action='store_true', default=False, help='only load model from checkpoint')
    parser.add_argument('--eval', action='store_true', default=False, help="evaluate only")
    parser.add_argument('--test', action='store_true', default=False, help="test only")
    parser.add_argument('--show', action='store_true', default=False, help="visualize the results")
    parser.add_argument('--log-steps', type=int, default=20, help="log every n steps to tensorboard")
    parser.add_argument('--save-steps', type=int, default=2000, help="save checkpoint every n steps")
    parser.add_argument('--accum-steps', type=int, default=1, help="update model weights every n steps")
    # misc
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=os.path.join(working_dir, '../temp', 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=os.path.join(working_dir, '../temp', 'logs'))
    parser.add_argument('--test-dir', type=str, metavar='PATH', default=os.path.join(working_dir, '../temp', 'test'))
    parser.add_argument('--model-path', type=str, metavar='PATH', default=None, help="pretrained model path")
    # distributed
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--net-card', type=str, default='', help="Name of the network card.")
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.net_card:
        os.environ['GLOO_SOCKET_IFNAME'] = args.net_card
        os.environ['NCCL_SOCKET_IFNAME'] = args.net_card

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # This enables tf32 on Ampere GPUs which is only 8% slower than
    # float16 and almost as accurate as float32
    # This was a default in pytorch until 1.12
    cuda.matmul.allow_tf32 = True
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False

    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.world_size > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size)
        dist.barrier()

    # Create loggers
    if not args.distributed or args.local_rank == 0:
        # Redirect print to both console and log file
        sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))
        # Create tensorboard writer
        tensorboard_path = os.path.join(args.logs_dir, 'tensorboard')
        os.makedirs(tensorboard_path, exist_ok=True)
        tb_writer = tensorboard.SummaryWriter(tensorboard_path)
    else:
        tb_writer = None

    # Create dataloaders
    data_root = os.path.join(args.data_dir, args.dataset.replace('_pipe', ''))

    train_transforms, train_collate_fn = transforms.create(
        args.dataset, train=True, root=data_root, version=args.version)
    test_transforms, test_collate_fn = transforms.create(
        args.dataset, train=False, root=data_root, version=args.version)

    if args.test_dataset is None:
        args.test_dataset = args.dataset

    train_dataset = datasets.create(
        args.dataset, data_root, split='train', transform=train_transforms, version=args.version)
    test_dataset = datasets.create(
        args.test_dataset, data_root, split='val', transform=test_transforms, version=args.version)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    train_loader = dataloaders.create(
        args.dataset, train_dataset, args.batch_size, train=True, distributed=args.distributed,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, collate_fn=train_collate_fn)
    test_loader = dataloaders.create(
        args.test_dataset, test_dataset, args.test_batch_size, train=False, distributed=args.distributed,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, collate_fn=test_collate_fn)

    # Create model
    norm_layer = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
    model = models.create(args.arch, norm_layer=norm_layer, num_classes=args.num_classes,
                          model_path=args.model_path, accum_steps=args.accum_steps)
    if not args.distributed or args.local_rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

    # Load from checkpoint
    checkpoint = dict()
    if args.resume:
        if args.eval or args.test:
            checkpoint = load_checkpoint(os.path.join(args.logs_dir, 'model_best.pth.tar'))
        else:
            checkpoint = load_checkpoint(os.path.join(args.logs_dir, 'checkpoint.pth.tar'))
        ret = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('missing_keys: {}'.format(ret.missing_keys))
        print('unexpected_keys: {}'.format(ret.unexpected_keys))

    # Put the model on GPU
    model = model.cuda()

    # Create optimizer
    if hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(lr=args.lr, weight_decay=args.weight_decay)
    else:
        param_groups = model.parameters()

    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=tuple(args.betas), eps=args.eps,
                                      weight_decay=args.weight_decay)
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, dampening=0,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
    else:
        raise NotImplementedError

    # Enable automated mixed precision
    if 'amp' in args.precision:
        grad_scaler = amp.GradScaler()
    else:
        grad_scaler = None

    if args.precision == 'amp':
        autocast = amp.autocast
    elif args.precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        autocast = amp.autocast(dtype=torch.bfloat16)
    else:
        autocast = suppress

    if args.schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=args.lr_min)
    elif args.schedule == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.gamma)
    else:
        raise NotImplementedError

    start_epoch = 0
    best_prec1 = 0
    is_best = False
    if args.resume and not args.load_model_only:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if grad_scaler is not None:
            grad_scaler.load_state_dict(checkpoint['grad_scaler'])
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        print("=> Start epoch {}  best_prec1 {:.4f}".format(start_epoch, best_prec1))

    # Parallelize the model
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            broadcast_buffers=True, find_unused_parameters=False)

    # Create Evaluator
    evaluator = Evaluator(args.task, model, autocast=autocast, tb_writer=tb_writer, num_classes=args.num_classes,
                          distributed=args.distributed)
    if args.eval:
        evaluator(test_loader)
        return

    # Create Tester
    tester = Tester(args.task, model, autocast=autocast, visualize=args.show, distributed=args.distributed, 
                    root=args.test_dir)
    if args.test:
        tester(test_loader)
        return

    # Create Trainer
    trainer = Trainer(model, optimizer, scheduler, grad_scaler, autocast=autocast, tb_writer=tb_writer, 
                      max_grad=args.max_grad, log_steps=args.log_steps, save_steps=args.save_steps,
                      accum_steps=args.accum_steps, distributed=args.distributed, root=args.logs_dir)

    # Start training
    for epoch in range(start_epoch, args.num_epochs):
        # Use .set_epoch() method to reshuffle the dataset partition at every iteration
        if hasattr(train_loader, 'set_epoch'):
            train_loader.set_epoch(epoch)

        trainer(train_loader, epoch, best_prec1)

        # evaluate on validation set
        if epoch >= args.eval_epoch:
            prec1 = evaluator(test_loader)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

        if not args.distributed or args.local_rank == 0:
            lr = scheduler.get_last_lr()
            print('epoch: {:d}, lr: {}'.format(epoch, lr))
            checkpoint = {
                'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_prec1': best_prec1,
            }
            if grad_scaler is not None:
                checkpoint['grad_scaler'] = grad_scaler.state_dict()
            save_checkpoint(checkpoint, is_best, fpath=os.path.join(args.logs_dir, 'checkpoint.pth.tar'))

        if args.distributed:
            dist.barrier()

    # Final test
    checkpoint = load_checkpoint(os.path.join(args.logs_dir, 'model_best.pth.tar'))
    if args.distributed:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    evaluator(test_loader)

    if hasattr(train_loader, 'shutdown'):
        train_loader.shutdown()
    if hasattr(test_loader, 'shutdown'):
        test_loader.shutdown()

    if hasattr(train_dataset, 'close'):
        train_dataset.close()
    if hasattr(test_dataset, 'close'):
        test_dataset.close()
    return


if __name__ == '__main__':
    parser = argument_parser()
    main(parser.parse_args())
