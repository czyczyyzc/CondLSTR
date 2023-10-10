import os
import time
import pickle
import torch
import torch.distributed as dist
from contextlib import suppress
from utils import Bar, to_device
from utils.meters import AverageMeter
from data.dataloaders.samplers.parquet_sampler import DistributedParquetSampler
from modeling import inferences


class Tester(object):
    def __init__(self, task, model, autocast=suppress, tb_writer=None, visualize=False, distributed=False, root=None):
        super(Tester, self).__init__()
        self.task        = task
        self.model       = model
        self.autocast    = autocast
        self.tb_writer   = tb_writer
        self.visualize   = visualize
        self.distributed = distributed
        self.root        = root
        self.inference   = inferences.create(task, visualize=visualize, root=root)

    def __call__(self, data_loader):
        self.model.eval()

        dataset = data_loader.dataset
        dataset_size = dataset.num_samples if hasattr(dataset, 'num_samples') else len(dataset)
        dataloader_size = data_loader.num_batches if hasattr(data_loader, 'num_batches') else len(data_loader)

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        bar = Bar('Processing', max=dataloader_size) if not self.distributed or dist.get_rank() == 0 else None

        results_list = []
        for i, data_dict in enumerate(data_loader):
            data_dict = to_device(data_dict, device='cuda', non_blocking=True)
            data_time.update(time.time() - end)
            end = time.time()

            with torch.no_grad(), self.autocast():
                pred_dict = self.model(data_dict)
                pred_list = self.inference(pred_dict, data_dict)
                results_list.extend(pred_list)

            batch_time.update(time.time() - end)
            end = time.time()
            if bar is not None:
                bar.suffix = "Testing: [{N_batch}/{N_size}] | Time data {N_dta:.3f} | " \
                             "Time batch {N_bta:.3f}".format(
                    N_batch=i+1, N_size=dataloader_size, N_dta=data_time.avg, N_bta=batch_time.avg)
                bar.next()
        if bar is not None:
            bar.finish()

        if self.distributed:
            dist.barrier()
            all_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_results, results_list)
            if isinstance(data_loader.batch_sampler, DistributedParquetSampler):
                results_list = sum(all_results, [])[: dataset_size]
            else:
                results_list = sum(map(list, zip(*all_results)), [])[: dataset_size]

        if not self.distributed or dist.get_rank() == 0:
            with open(os.path.join(self.root, 'results.pkl'), 'wb') as f:
                pickle.dump(results_list, f)
        return
