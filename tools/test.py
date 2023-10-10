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
from engine.tester import Tester
from engine.evaluator import Evaluator
from utils.logger import Logger
from utils.serialization import load_checkpoint
from utils.torch2jit import torch2jit
from utils.flop_count import flop_count
# from utils.torch2tRt import torch2trt
# from utils.torch2onnx import torch2onnx
# from utils.onnx2tRt import onnx2trt

import warnings
warnings.filterwarnings('ignore')


def argument_parser():
    parser = argparse.ArgumentParser(description='HDMapNet with Pytorch Implementation')
    parser.add_argument('--gpu-ids', type=str, default='0')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='laion_sa1b', choices=datasets.names())
    parser.add_argument('-v', '--version', type=str, default='v1.0')
    parser.add_argument('-j', '--num-workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', nargs='+', type=int, default=[4])
    parser.add_argument('-pf', '--prefetch-factor', type=int, default=2, help="batches prefetched by each worker")
    parser.add_argument('--split', type=str, default='val', help="The split of the dataset")
    # model
    parser.add_argument('-a', '--arch', type=str, default='SamClipMix', choices=models.names())
    parser.add_argument('-t', '--task', type=str, default='segmentation', help="The task to carry out")
    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['amp', 'amp_bf16', 'fp32'],
                        help="floating point precision")
    parser.add_argument('-c', '--num-classes', type=int, default=1, help="number of classes")
    # testing
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--eval', action='store_true', default=False, help="evaluate only")
    parser.add_argument('--show', action='store_true', default=False, help="visualize the results")
    parser.add_argument('--flop', action='store_true', default=False, help="count the flops of the model")
    parser.add_argument('--onnx', action='store_true', default=False, help="pytorch to onnx")
    parser.add_argument('--trt', action='store_true', default=False, help="onnx to tensorrt")
    parser.add_argument('--jit', action='store_true', default=False, help='apply jit trace to the model')
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

    test_transforms, test_collate_fn = transforms.create(
        args.dataset, train=False, root=data_root, version=args.version)

    test_dataset = datasets.create(
        args.dataset, data_root, split=args.split, transform=test_transforms, version=args.version)

    test_loader = dataloaders.create(
        args.dataset, test_dataset, args.batch_size, train=False, distributed=args.distributed,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, collate_fn=test_collate_fn)

    # Create model
    norm_layer = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
    model = models.create(args.arch, norm_layer=norm_layer, num_classes=args.num_classes,
                          model_path=args.model_path).eval()
    if not args.distributed or args.local_rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

    # Load from checkpoint
    model_path = os.path.join(args.logs_dir, 'model_best.pth.tar')
    if os.path.exists(model_path):
        print("Loading model from {} ...".format(model_path))
        checkpoint = load_checkpoint(model_path)
        ret = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print('missing_keys: {}'.format(ret.missing_keys))
        print('unexpected_keys: {}'.format(ret.unexpected_keys))
    else:
        print("Model path {} does not exist!".format(model_path))

    if args.flop:
        input_dict = test_dataset.__getitem__(0)
        flop_count(model, input_dict)
        sys.exit()

    if args.jit:
        jit_file = os.path.join(args.logs_dir, 'model_best.jit')
        input_dict = test_dataset.__getitem__(0)
        torch2jit(
            model,
            input_dict,
            output_file=jit_file,
            verify=True
        )
        sys.exit()

    if args.onnx:
        jit_file = os.path.join(args.logs_dir, 'model_best.jit')
        # assert os.path.exists(jit_file), "Have to generate jit model file first!"
        onnx_file = os.path.join(args.logs_dir, 'model_best.onnx')
        input_dict = test_dataset.__getitem__(0)
        torch2onnx(
            model,
            jit_file,
            onnx_file,
            input_dict,
            opset_version=11,
            do_simplify=True,
            dynamic_export=False,
            do_constant_folding=False,
            verify=False,
            verbose=True
        )
        sys.exit()

    if args.trt:
        onnx_file = os.path.join(args.logs_dir, 'model_best.onnx')
        assert os.path.exists(onnx_file), "Have to generate onnx model file first!"
        trt_file = os.path.join(args.logs_dir, 'model_best.trt')
        input_dict = test_dataset.__getitem__(0)
        onnx2trt(
            model,
            onnx_file,
            trt_file,
            input_dict,
            fp16_mode=False,
            verify=False,
            verbose=True
        )
        # trt_file = os.path.join(args.logs_dir, 'model_best.trt')
        # input_dict = test_dataset.__getitem__(0)
        # torch2trt(
        #     model,
        #     trt_file,
        #     input_dict,
        #     fp16_mode=False,
        #     workspace_size=2,
        #     verify=True
        # )
        sys.exit()

    # Put the model on GPU
    model = model.cuda()

    # Parallelize the model
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    if args.precision == 'amp':
        autocast = amp.autocast
    elif args.precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        autocast = amp.autocast(dtype=torch.bfloat16)
    else:
        autocast = suppress

    # Create Evaluator
    evaluator = Evaluator(args.task, model, autocast=autocast, tb_writer=tb_writer, num_classes=args.num_classes,
                          distributed=args.distributed)
    if args.eval:
        evaluator(test_loader)
        return

    # Create Tester
    tester = Tester(args.task, model, autocast=autocast, tb_writer=tb_writer, visualize=args.show,
                    distributed=args.distributed, root=args.test_dir)
    tester(test_loader)

    if hasattr(test_loader, 'shutdown'):
        test_loader.shutdown()

    if hasattr(test_dataset, 'close'):
        test_dataset.close()
    return


if __name__ == '__main__':
    parser = argument_parser()
    main(parser.parse_args())
