import time
import torch
import torch.distributed as dist
from contextlib import suppress
from utils import Bar, to_device
from utils.meters import AverageMeter
from modeling import metrics


class Evaluator(object):
    def __init__(self, task, model, autocast=suppress, tb_writer=None, num_classes=1, distributed=False):
        super(Evaluator, self).__init__()
        self.task        = task
        self.model       = model
        self.autocast    = autocast
        self.tb_writer   = tb_writer
        self.num_classes = num_classes
        self.distributed = distributed
        self.metric      = metrics.create(task, num_classes=num_classes)

    def __call__(self, data_loader):
        self.model.train(False)

        dataloader_size = data_loader.num_batches if hasattr(data_loader, 'num_batches') else len(data_loader)

        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        bar = Bar('Processing', max=dataloader_size) if not self.distributed or dist.get_rank() == 0 else None
        for i, data_dict in enumerate(data_loader):
            data_dict = to_device(data_dict, device='cuda', non_blocking=True)
            data_time.update(time.time() - end)
            end = time.time()

            with torch.no_grad(), self.autocast():
                pred_dict = self.model(data_dict)
                self.metric.add_batch(pred_dict, data_dict)

            batch_time.update(time.time() - end)
            end = time.time()
            if bar is not None:
                bar.suffix = "Evaluating: [{N_batch}/{N_size}] | Time data {T_data:.3f} | " \
                             "Time batch {T_batch:.3f}".format(
                    N_batch=i+1, N_size=dataloader_size, T_data=data_time.avg, T_batch=batch_time.avg)
                bar.next()
        if bar is not None:
            bar.finish()

        if self.distributed:
            state = torch.from_numpy(self.metric.dump_state()).cuda()
            dist.reduce(state, dst=0)
            self.metric.load_state(state.cpu().numpy())

        eval_dict = self.metric()
        if not self.distributed or dist.get_rank() == 0:
            print_info = 'Result:'
            for key, value in eval_dict.items():
                print_info += ' ' + key + ': ' + '{N_acc:.3f}'.format(N_acc=value)
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('val/' + key, value)
            print(print_info)
        return eval_dict['avg']

