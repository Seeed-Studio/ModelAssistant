# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from typing import Iterator, List, Optional, Union

import torch
import random
import numpy as np
from torch.utils.data import ConcatDataset
from mmengine.dataset import DefaultSampler

from sscma.registry import DATA_SANPLERS


@DATA_SANPLERS.register_module()
class SemiSampler(DefaultSampler):
    """
    Sampler for scaled sampling of semi-supervised data

    Params:
        dataset (torch::ConcatDataset): Multiple merged datasets
        batch_size (int): Training is set batch_size
        sample_ratio (List[int,float]): Sampling rate for each dataset
            with length equal to the number of datasets in dataset
        shuffle: (bool): Whether to disrupt the sampling order of the data
        seed (int): Random seed used
        round_up (bool): If the ratio of the length of each dataset is not
            equal to the ratio of the sampling rate, whether to resample
            the under-sampled data.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int,
        sample_ratio: List[Union[int, float]],
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = False,
    ) -> None:
        assert len(sample_ratio) == len(
            dataset.cumulative_sizes
        ), "Sampling rate length must be equal to the number of datasets."

        super(SemiSampler, self).__init__(dataset, shuffle=shuffle, seed=seed, round_up=round_up)
        if seed is not None:
            self.set_seed(seed)

        self.dataset = dataset
        self._batch_size = batch_size
        self.round_up = round_up
        self.shuffle = shuffle
        sample_ratio = [scale / sum(sample_ratio) for scale in sample_ratio]
        self.sample_ratio = sample_ratio

        data_index = np.arange(dataset.cumulative_sizes[1])
        self.data1 = data_index[: dataset.cumulative_sizes[0]]
        self.data2 = data_index[dataset.cumulative_sizes[0] :]
        self.computer_sampler_size()
        # Whether to load unlabeled data
        self._only_label = True
        self._only_unlabel = False
        self._all_data = False
        self.labels = ['_only_label', '_only_unlabel', '_all_data']
        self.datasets_len = dataset.cumulative_sizes[-1]

        self.computer_epoch()

    def __iter__(self) -> Iterator[int]:
        indexs = []
        num1 = 0
        num2 = 0
        data1_len = len(self.data1)
        data2_len = len(self.data2)
        if self.shuffle:
            np.random.shuffle(self.data1)
            np.random.shuffle(self.data2)

        for _ in range(self.total_epoch):
            if self.all_data:
                for _ in range(self.sample_size[0]):
                    indexs.append(self.data1[num1 % data1_len])
                    num1 += 1
                for _ in range(self.sample_size[1]):
                    indexs.append(self.data2[num2 % data2_len])
                    num2 += 1
            elif self.only_label:
                for _ in range(self.sample_size[0] + self.sample_size[1]):
                    indexs.append(self.data1[num1 % data1_len])
                    num1 += 1
            else:
                for _ in range(self.sample_size[0] + self.sample_size[1]):
                    indexs.append(self.data2[num2 % data2_len])
                    num2 += 1

        return iter(indexs)

    def __len__(self) -> int:
        return self.total_epoch * self._batch_size

    def computer_sampler_size(self):
        frist_size = int(self.sample_ratio[0] * self.batch_size)
        if frist_size >= self.batch_size:
            frist_size = self.batch_size - 1
        elif frist_size <= 0:
            frist_size = 1

        self.sample_size = [frist_size, self.batch_size - frist_size]

    def computer_epoch(self):
        if self.all_data:
            if self.round_up:
                self.total_epoch = max(len(self.data1) // self.sample_size[0], len(self.data2) // self.sample_size[1])
            else:
                self.total_epoch = min(len(self.data1) // self.sample_size[0], len(self.data2) // self.sample_size[1])
        elif self.only_label:
            self.total_epoch = len(self.data1) // sum(self.sample_size)
        else:
            self.total_epoch = len(self.data2) // sum(self.sample_size)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch: int) -> None:
        self._batch_size = batch
        self.computer_sampler_size()

    def set_seed(self, seed: int) -> None:
        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def set_label(self, attr: str, flag):
        for i in self.labels:
            if attr in i:
                setattr(self, i, flag)
            else:
                setattr(self, i, not flag)
        self.computer_epoch()

    @property
    def only_label(self):
        return self._only_label

    @only_label.setter
    def only_label(self, flag: bool):
        self.set_label('only_label', flag)

    @property
    def only_unlabel(self):
        return self._only_unlabel

    @only_unlabel.setter
    def only_unlabel(self, flag: bool):
        self.set_label('only_unlabel', flag)

    @property
    def all_data(self):
        return self._all_data

    @all_data.setter
    def all_data(self, flag: bool):
        self.set_label('all_data', flag)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
