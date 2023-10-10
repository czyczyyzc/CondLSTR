import os
import time
import torch.nn as nn
import torch.distributed as dist
from contextlib import suppress
from collections import defaultdict
from utils import Bar, to_device
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint


class Trainer(object):
    def __init__(self, model, optimizer, scheduler, grad_scaler=None, autocast=suppress, tb_writer=None,
                 max_grad=None, log_steps=20, save_steps=2000, accum_steps=1, distributed=False, root=None):
        super(Trainer, self).__init__()
        self.model       = model
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.grad_scaler = grad_scaler
        self.autocast    = autocast
        self.tb_writer   = tb_writer
        self.max_grad    = max_grad
        self.log_steps   = log_steps
        self.save_steps  = save_steps
        self.accum_steps = accum_steps
        self.distributed = distributed
        self.root        = root

    def __call__(self, data_loader, epoch, best_prec1):
        self.model.train()
        self.optimizer.zero_grad()

        dataloader_size = data_loader.num_batches if hasattr(data_loader, 'num_batches') else len(data_loader)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_avg = AverageMeter()
        loss_dict_avg = defaultdict(AverageMeter)

        end = time.time()
        bar = Bar('Processing', max=dataloader_size) if not self.distributed or dist.get_rank() == 0 else None
        step = 0
        for data_dict in data_loader:
            data_dict = to_device(data_dict, device='cuda', non_blocking=True)
            data_time.update(time.time() - end)
            end = time.time()
            
            with self.autocast():
                if self.distributed:
                    if hasattr(self.model.module, 'forward_accum'):
                        if not self.model.module.forward_accum(data_dict):
                            continue
                else:
                    if hasattr(self.model, 'forward_accum'):
                        if not self.model.forward_accum(data_dict):
                            continue
            
            with self.autocast():
                loss_dict = self.model(data_dict)
                loss = sum([v for k, v in loss_dict.items()]) / self.accum_steps

            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.accum_steps == 0:
                if self.grad_scaler is not None:
                    if self.max_grad is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad, norm_type=2.0)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad()
                else:
                    if self.max_grad is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad, norm_type=2.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            step = step + 1
            loss = loss.detach().cpu().numpy() * self.accum_steps
            loss_avg.update(loss, len(data_dict['img']))
            for k, v in loss_dict.items():
                loss_dict_avg[k].update(v.detach().cpu().numpy(), len(data_dict['img']))

            if not self.distributed or dist.get_rank() == 0:
                # write to tensorboard
                if self.tb_writer is not None and step % self.log_steps == 0:
                    global_step = epoch * dataloader_size + step
                    self.tb_writer.add_scalar('train/loss', loss_avg.val, global_step)
                    for k, v in loss_dict_avg.items():
                        global_step = epoch * dataloader_size + step
                        self.tb_writer.add_scalar('train/' + k, v.val, global_step)
                # save checkpoint
                if self.root is not None and step % self.save_steps == 0:
                    is_best = False
                    checkpoint = {
                        'state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'epoch': epoch,
                        'best_prec1': best_prec1,
                    }
                    if self.grad_scaler is not None:
                        checkpoint['grad_scaler'] = self.grad_scaler.state_dict()
                    save_checkpoint(checkpoint, is_best, fpath=os.path.join(self.root, 'checkpoint.pth.tar'))

            batch_time.update(time.time() - end)
            end = time.time()
            if bar is not None:
                bar.suffix = "Epoch: [{N_epoch}][{N_batch}/{N_size}] | " \
                             "T_data {T_data:.3f} | T_batch {T_batch:.3f} | " \
                             "Loss {N_loss:.3f}".format(
                    N_epoch=epoch, N_batch=step, N_size=dataloader_size,
                    T_data=data_time.avg, T_batch=batch_time.avg, N_loss=loss_avg.avg,
                )
                for key, value in loss_dict_avg.items():
                    bar.suffix += ' | ' + key + ' {N_loss:.3f}'.format(N_loss=value.avg)
                bar.next()

        self.scheduler.step()
        if bar is not None:
            bar.finish()
        return
