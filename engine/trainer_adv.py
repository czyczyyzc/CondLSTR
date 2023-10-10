import os
import time
import torch.nn as nn
import torch.distributed as dist
from contextlib import suppress
from collections import defaultdict
from utils import Bar, to_device, toggle_grad
from utils.meters import AverageMeter
from utils.serialization import save_checkpoint


class Trainer(object):
    def __init__(self, model, optimizer_G, optimizer_D, scheduler_G, scheduler_D, grad_scaler_G=None, grad_scaler_D=None,
                 autocast=suppress, tb_writer=None, max_grad_G=None, max_grad_D=None, num_steps_D=1,
                 log_steps=20, save_steps=2000, accum_steps=1, distributed=False, root=None):
        super(Trainer, self).__init__()
        self.model         = model
        self.optimizer_G   = optimizer_G
        self.optimizer_D   = optimizer_D
        self.scheduler_G   = scheduler_G
        self.scheduler_D   = scheduler_D
        self.grad_scaler_G = grad_scaler_G
        self.grad_scaler_D = grad_scaler_D
        self.autocast      = autocast
        self.tb_writer     = tb_writer
        self.max_grad_G    = max_grad_G
        self.max_grad_D    = max_grad_D
        self.num_steps_D   = num_steps_D
        self.log_steps     = log_steps
        self.save_steps    = save_steps
        self.accum_steps   = accum_steps
        self.distributed   = distributed
        self.root          = root

    def __call__(self, data_loader, epoch, best_prec1):
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()

        dataloader_size = data_loader.num_batches if hasattr(data_loader, 'num_batches') else len(data_loader)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_avg = AverageMeter()
        loss_dict_avg = defaultdict(AverageMeter)

        end = time.time()
        bar = Bar('Processing', max=dataloader_size) if not self.distributed or dist.get_rank() == 0 else None
        step = 0
        data_list = []
        for data_dict in data_loader:
            if len(data_list) < self.num_steps_D + 1:
                data_dict = to_device(data_dict, device='cuda', non_blocking=True)
                data_list.append(data_dict)
                continue
            
            data_time.update(time.time() - end)
            end = time.time()

            # Optimize discriminator
            loss_dict_D = {}
            loss_D = 0
            for _ in range(self.num_steps_D):
                toggle_grad(self.model.discriminator, True)
                toggle_grad(self.model.generator, False)

                self.model.discriminator.train()
                self.model.generator.eval()

                with self.autocast():
                    if hasattr(self.model, 'forward_accum_D'):
                        if not self.model.forward_accum_D(data_dict):
                            continue
                
                with self.autocast():
                    loss_dict_D = self.model.forward_D(data_list.pop())
                    loss_D = sum([v for k, v in loss_dict_D.items()]) / self.accum_steps

                if self.grad_scaler_D is not None:
                    self.grad_scaler_D.scale(loss_D).backward()
                else:
                    loss_D.backward()

                if ((step + 1) * self.num_steps_D) % self.accum_steps == 0:
                    if self.grad_scaler_D is not None:
                        if self.max_grad_D is not None:
                            self.grad_scaler_D.unscale_(self.optimizer_D)
                            nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.max_grad_D, norm_type=2.0)
                        self.grad_scaler_D.step(self.optimizer_D)
                        self.grad_scaler_D.update()
                        self.optimizer_D.zero_grad()
                    else:
                        if self.max_grad_D is not None:
                            nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), self.max_grad_D, norm_type=2.0)
                        self.optimizer_D.step()
                        self.optimizer_D.zero_grad()
            
            # Optimize generator
            toggle_grad(self.model.discriminator, False)
            toggle_grad(self.model.generator, True)

            self.model.discriminator.eval()
            self.model.generator.train()

            with self.autocast():
                if hasattr(self.model, 'forward_accum_G'):
                    if not self.model.forward_accum_G(data_dict):
                        continue
            
            with self.autocast():
                loss_dict_G = self.model.forward_G(data_list.pop())
                loss_G = sum([v for k, v in loss_dict_G.items()]) / self.accum_steps

            if self.grad_scaler_G is not None:
                self.grad_scaler_G.scale(loss_G).backward()
            else:
                loss_G.backward()

            if (step + 1) % self.accum_steps == 0:
                if self.grad_scaler_G is not None:
                    if self.max_grad_G is not None:
                        self.grad_scaler_G.unscale_(self.optimizer_G)
                        nn.utils.clip_grad_norm_(self.model.generator.parameters(), self.max_grad_G, norm_type=2.0)
                    self.grad_scaler_G.step(self.optimizer_G)
                    self.grad_scaler_G.update()
                    self.optimizer_G.zero_grad()
                else:
                    if self.max_grad_G is not None:
                        nn.utils.clip_grad_norm_(self.model.generator.parameters(), self.max_grad_G, norm_type=2.0)
                    self.optimizer_G.step()
                    self.optimizer_G.zero_grad()
            
            step = step + 1
            loss = loss_G + loss_D
            loss = loss.detach().cpu().numpy() * self.accum_steps
            loss_avg.update(loss, len(data_dict['img']))
            for k, v in loss_dict_G.items():
                loss_dict_avg[k].update(v.detach().cpu().numpy(), len(data_dict['img']))
            for k, v in loss_dict_D.items():
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
                        'state_dict_G': self.model.generator.module.state_dict() \
                            if self.distributed else self.model.generator.state_dict(),
                        'state_dict_D': self.model.discriminator.module.state_dict() \
                            if self.distributed else self.model.discriminator.state_dict(),
                        'optimizer_G': self.optimizer_G.state_dict(),
                        'optimizer_D': self.optimizer_D.state_dict(),
                        'scheduler_G': self.scheduler_G.state_dict(),
                        'scheduler_D': self.scheduler_D.state_dict(),
                        'epoch': epoch,
                        'best_prec1': best_prec1,
                    }
                    if self.grad_scaler_G is not None:
                        checkpoint['grad_scaler_G'] = self.grad_scaler_G.state_dict()
                    if self.grad_scaler_D is not None:
                        checkpoint['grad_scaler_D'] = self.grad_scaler_D.state_dict()
                    save_checkpoint(checkpoint, is_best, fpath=os.path.join(self.root, 'checkpoint.pth.tar'))

            batch_time.update(time.time() - end)
            end = time.time()
            if bar is not None:
                bar.suffix = "Epoch: [{N_epoch}][{N_batch}/{N_size}] | " \
                             "Time data {T_data:.3f} | Time batch {T_batch:.3f} | " \
                             "Loss all {N_loss:.3f}".format(
                    N_epoch=epoch, N_batch=step, N_size=dataloader_size,
                    T_data=data_time.avg, T_batch=batch_time.avg, N_loss=loss_avg.avg,
                )
                for key, value in loss_dict_avg.items():
                    bar.suffix += ' | ' + key + ' {N_loss:.3f}'.format(N_loss=value.avg)
                bar.next()

        self.scheduler_G.step()
        self.scheduler_D.step()
        if bar is not None:
            bar.finish()
        return
