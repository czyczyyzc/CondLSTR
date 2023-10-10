import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, \
    SequentialSampler, default_collate
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, \
    DistributedReadingService, SequentialReadingService
from .samplers import *


def simple_dataloader(dataset, batch_size, train=True, distributed=False, num_workers=4, prefetch_factor=2,
                      collate_fn=default_collate, **kwargs):
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        if train:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

    if isinstance(batch_size, list):
        batch_size = batch_size[0]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, prefetch_factor=prefetch_factor,
        pin_memory=True, drop_last=train,  collate_fn=collate_fn)

    if distributed:
        dataloader.set_epoch = sampler.set_epoch
    return dataloader


def parquet_dataloader(dataset, batch_size, train=True, distributed=False, num_workers=4, prefetch_factor=2,
                       collate_fn=default_collate, **kwargs):
    if isinstance(batch_size, list):
        batch_size = batch_size[0]
    if distributed:
        batch_sampler = DistributedParquetSampler(dataset, batch_size, shuffle=train, drop_last=train)
    else:
        batch_sampler = ParquetSampler(dataset, batch_size, shuffle=train, drop_last=train)

    dataloader = DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=num_workers, prefetch_factor=prefetch_factor,
        pin_memory=True, collate_fn=collate_fn)

    if distributed:
        dataloader.set_epoch = batch_sampler.set_epoch
    return dataloader


def mix_parquet_dataloader(dataset, batch_size, train=True, distributed=False, num_workers=4, prefetch_factor=2,
                           collate_fn=default_collate, **kwargs):
    if isinstance(batch_size, int):
        batch_size = [batch_size]
    if distributed:
        batch_sampler = DistributedMixParquetSampler(dataset, batch_size, shuffle=train, drop_last=train)
    else:
        batch_sampler = MixParquetSampler(dataset, batch_size, shuffle=train, drop_last=train)

    dataloader = DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=num_workers, prefetch_factor=prefetch_factor,
        pin_memory=True, collate_fn=collate_fn)

    if distributed:
        dataloader.set_epoch = batch_sampler.set_epoch
    return dataloader


def pipe_dataloader(datapipe, batch_size, train=True, distributed=False, num_workers=4, prefetch_factor=2,
                    collate_fn=default_collate, **kwargs):
    if isinstance(batch_size, list):
        batch_size = batch_size[0]

    num_samples = datapipe.num_samples
    datapipe = datapipe.batch(batch_size, drop_last=train)
    datapipe = datapipe.collate(collate_fn=collate_fn)
    datapipe = datapipe.prefetch(buffer_size=prefetch_factor)
    datapipe = datapipe.pin_memory()

    if distributed:
        mp_rs = MultiProcessingReadingService(num_workers=num_workers)
        dist_rs = DistributedReadingService()
        rs = SequentialReadingService(dist_rs, mp_rs)
        dataloader = DataLoader2(datapipe, reading_service=rs)
    else:
        rs = MultiProcessingReadingService(num_workers=num_workers)
        dataloader = DataLoader2(datapipe, reading_service=rs)

    num_replicas = 1
    if distributed:
        num_replicas = dist.get_world_size()
    num_samples_per_worker = num_samples // (num_replicas * num_workers)
    if train:
        num_batches_per_worker = num_samples_per_worker // batch_size
    else:
        num_batches_per_worker = (num_samples_per_worker + batch_size - 1) // batch_size
    num_batches = num_batches_per_worker * num_workers

    datapipe.num_samples = num_samples
    dataloader.dataset = datapipe
    dataloader.set_epoch = dataloader.seed
    dataloader.num_batches = num_batches
    return dataloader
