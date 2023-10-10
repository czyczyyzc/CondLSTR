import copy
import math
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from typing import Iterator, Optional, List, Sized, TypeVar

T_co = TypeVar('T_co', covariant=True)


class MixParquetSampler(Sampler[List[int]]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (list): list of size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, batch_size: list, shuffle: bool = False, drop_last: bool = False,
                 replacement: bool = False, num_samples: Optional[int] = None, generator=None) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, list) or any([isinstance(x, bool) for x in batch_size]) or \
                any([x <= 0 for x in batch_size]):
            raise ValueError("batch_size should be a list of positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.data_index_list = copy.deepcopy(self.data_source.data_index_list)
        del self.data_source.data_index_list

        assert len(self.batch_size) == len(self.data_index_list), \
            "The number of batch_sizes {} and the number of datasets {} do not match".format(
                len(self.batch_size), len(self.data_index_list))

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[List[int]]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # n = len(self.data_source)
        # if self.replacement:
        #     for _ in range(self.num_samples // 32):
        #         yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
        #     yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        # else:
        #     for _ in range(self.num_samples // n):
        #         yield from torch.randperm(n, generator=generator).tolist()
        #     yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

        data_index_list = copy.deepcopy(self.data_index_list)
        if self.shuffle:
            for i, file_index_list in enumerate(data_index_list):
                # shuffle files
                index = torch.randperm(len(file_index_list), generator=generator).tolist()
                file_index_list = [file_index_list[i] for i in index]
                # shuffle data in each file
                for j, data_index in enumerate(file_index_list):
                    index = torch.randperm(len(data_index), generator=generator).numpy()
                    file_index_list[j] = data_index[index]
                data_index_list[i] = file_index_list

        data_batch_list = []
        for i, file_index_list in enumerate(data_index_list):
            file_batch_list = []
            for j, data_index in enumerate(file_index_list):
                num_batches = len(data_index) // self.batch_size[i]
                batches = data_index[: num_batches * self.batch_size[i]].reshape(num_batches, self.batch_size[i]).tolist()
                if not self.drop_last and (len(data_index) > num_batches * self.batch_size[i]):
                    batches.append(data_index[num_batches * self.batch_size[i]:].tolist())
                file_batch_list.extend(batches)
            data_batch_list.append(file_batch_list)

        num_batches_min = min([len(file_batch_list) for file_batch_list in data_batch_list])
        data_batch_list = [file_batch_list[:num_batches_min] for file_batch_list in data_batch_list]
        batch_list = [sum(batches, []) for batches in list(zip(*data_batch_list))]
        return iter(batch_list)

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return min([sum([len(index) // self.batch_size[i] for index in file_index_list])
                        for i, file_index_list in enumerate(self.data_index_list)])
        else:
            return min([sum([(len(index) + self.batch_size[i] - 1) // self.batch_size[i] for index in file_index_list])
                        for i, file_index_list in enumerate(self.data_index_list)])


class DistributedMixParquetSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        batch_size (list): list of size of mini-batch.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, batch_size: list, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, list) or any([isinstance(x, bool) for x in batch_size]) or \
                any([x <= 0 for x in batch_size]):
            raise ValueError("batch_size should be a list of positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.data_index_list = copy.deepcopy(self.dataset.data_index_list)
        del self.dataset.data_index_list

        assert len(self.batch_size) == len(self.data_index_list), \
            "The number of batch_sizes {} and the number of datasets {} do not match".format(
                len(self.batch_size), len(self.data_index_list))

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
        #     # Split to nearest available length that is evenly divisible.
        #     # This is to ensure each rank receives the same amount of data when
        #     # using this Sampler.
        #     self.num_samples = math.ceil(
        #         (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
        #     )
        # else:
        #     self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        # self.total_size = self.num_samples * self.num_replicas

        if self.drop_last:
            num_batches = min([sum([len(index) // self.batch_size[i] for index in file_index_list])
                               for i, file_index_list in enumerate(self.data_index_list)])
            if num_batches % self.num_replicas != 0:
                num_batches = math.ceil(
                    (num_batches - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
                )
            else:
                num_batches = num_batches // self.num_replicas
        else:
            num_batches = min([sum([(len(index) + self.batch_size[i] - 1) // self.batch_size[i] for index in file_index_list])
                               for i, file_index_list in enumerate(self.data_index_list)])
            num_batches = math.ceil(num_batches / self.num_replicas)  # type: ignore[arg-type]
        self.num_batches = num_batches
        self.total_size = self.num_batches * self.num_replicas

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        # if self.shuffle:
        #     # deterministically shuffle based on epoch and seed
        #     g = torch.Generator()
        #     g.manual_seed(self.seed + self.epoch)
        #     indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        # else:
        #     indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        data_index_list = copy.deepcopy(self.data_index_list)
        if self.shuffle:
            for i, file_index_list in enumerate(data_index_list):
                # shuffle files
                index = torch.randperm(len(file_index_list), generator=g).tolist()
                file_index_list = [file_index_list[i] for i in index]
                # shuffle data in each file
                for j, data_index in enumerate(file_index_list):
                    index = torch.randperm(len(data_index), generator=g).numpy()
                    file_index_list[j] = data_index[index]
                data_index_list[i] = file_index_list

        data_batch_list = []
        for i, file_index_list in enumerate(data_index_list):
            file_batch_list = []
            for j, data_index in enumerate(file_index_list):
                num_batches = len(data_index) // self.batch_size[i]
                batches = data_index[: num_batches * self.batch_size[i]].reshape(num_batches, self.batch_size[i]).tolist()
                if not self.drop_last and (len(data_index) > num_batches * self.batch_size[i]):
                    batches.append(data_index[num_batches * self.batch_size[i]:].tolist())
                file_batch_list.extend(batches)
            data_batch_list.append(file_batch_list)

        num_batches_min = min([len(file_batch_list) for file_batch_list in data_batch_list])
        data_batch_list = [file_batch_list[:num_batches_min] for file_batch_list in data_batch_list]
        batch_list = [sum(batches, []) for batches in list(zip(*data_batch_list))]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(batch_list)
            if padding_size <= len(batch_list):
                batch_list += batch_list[:padding_size]
            else:
                batch_list += (batch_list * math.ceil(padding_size / len(batch_list)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            batch_list = batch_list[:self.total_size]
        assert len(batch_list) == self.total_size

        # subsample
        # batch_list = batch_list[self.rank:self.total_size:self.num_replicas]
        batch_list = batch_list[self.rank * self.num_batches: (self.rank + 1) * self.num_batches]
        assert len(batch_list) == self.num_batches

        return iter(batch_list)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
