# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial

import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import TORCH_VERSION, Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader
from mmdet.datasets.builder import worker_init_fn
from mmdet.datasets.samplers import (
    ClassAwareSampler,
    DistributedGroupSampler,
    DistributedSampler,
    GroupSampler,
    InfiniteBatchSampler,
    InfiniteGroupBatchSampler,
)


def collate_fn(batch):
    img, label = [x['img'] for x in batch], [y['target'] for y in batch]
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return dict(img=torch.stack(img), target=torch.cat(label, 0))


def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dist=True,
    shuffle=True,
    seed=None,
    runner_type='EpochBasedRunner',
    persistent_workers=False,
    class_aware_sampler=None,
    **kwargs,
):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int, Optional): Seed to be used. Default: None.
        runner_type (str): Type of runner. Default: `EpochBasedRunner`
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers `Dataset` instances alive.
            This argument is only valid when PyTorch>=1.7.0. Default: False.
        class_aware_sampler (dict): Whether to use `ClassAwareSampler`
            during training. Default: None.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    if dist:
        # When model is :obj:`DistributedDataParallel`,
        # `batch_size` of :obj:`dataloader` is the
        # number of training samples on each GPU.
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # When model is obj:`DataParallel`
        # the batch size is samples on all the GPUS
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    if runner_type == 'IterBasedRunner':
        # this is a batch sampler, which can yield
        # a mini-batch indices each time.
        # it can be used in both `DataParallel` and
        # `DistributedDataParallel`
        if shuffle:
            batch_sampler = InfiniteGroupBatchSampler(dataset, batch_size, world_size, rank, seed=seed)
        else:
            batch_sampler = InfiniteBatchSampler(dataset, batch_size, world_size, rank, seed=seed, shuffle=False)
        batch_size = 1
        sampler = None
    else:
        if class_aware_sampler is not None:
            # ClassAwareSampler can be used in both distributed and
            # non-distributed training.
            num_sample_class = class_aware_sampler.get('num_sample_class', 1)
            sampler = ClassAwareSampler(
                dataset, samples_per_gpu, world_size, rank, seed=seed, num_sample_class=num_sample_class
            )
        elif dist:
            # DistributedGroupSampler will definitely shuffle the data to
            # satisfy that images on each GPU are in the same group
            if shuffle:
                sampler = DistributedGroupSampler(dataset, samples_per_gpu, world_size, rank, seed=seed)
            else:
                sampler = DistributedSampler(dataset, world_size, rank, shuffle=False, seed=seed)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_sampler = None

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None

    if TORCH_VERSION != 'parrots' and digit_version(TORCH_VERSION) >= digit_version('1.7.0'):
        kwargs['persistent_workers'] = persistent_workers
    elif persistent_workers is True:
        warnings.warn('persistent_workers is invalid because your pytorch ' 'version is lower than 1.7.0')

    collate_ = collate_fn if 'collate' in kwargs else partial(collate, samples_per_gpu=samples_per_gpu)
    kwargs.pop('collate') if 'collate' in kwargs else None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collate_,
        pin_memory=kwargs.pop('pin_memory', False),
        worker_init_fn=init_fn,
        **kwargs,
    )

    return data_loader

    return data_loader
