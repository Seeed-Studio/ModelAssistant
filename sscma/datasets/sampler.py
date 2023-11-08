from typing import Iterator, List, Optional, Any, Union
from mmengine.dataset import DefaultSampler
from mmengine.dataset.sampler import DefaultSampler
from sscma.registry import DATA_SANPLERS

from torch.utils.data import ConcatDataset, DataLoader, Dataset

import numpy as np
import torch
import random


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
        self.sample_size = [int(sample_ratio[0] * batch_size), batch_size - int(sample_ratio[0] * batch_size)]

        data_index = np.arange(dataset.cumulative_sizes[1])
        self.data1 = data_index[: dataset.cumulative_sizes[0]]
        self.data2 = data_index[dataset.cumulative_sizes[0] :]

        # 是否加载无标签数据
        self._with_unlabel = False
        self.datasets_len = dataset.cumulative_sizes[-1]
        self.computer_epoch()

    def __iter__(self) -> Iterator[int]:
        indexs = []
        num1 = 0
        num2 = 0
        if self.shuffle:
            np.random.shuffle(self.data1)
            np.random.shuffle(self.data2)

        for i in range(self.total_epoch):
            if self.with_unlabel:
                for _ in range(self.sample_size[0]):
                    indexs.append(self.data1[num1 % len(self.data1)])
                    num1 += 1
                for _ in range(self.sample_size[1]):
                    indexs.append(self.data2[num2 % len(self.data2)])
                    num2 += 1
            else:
                for _ in range(self.sample_size[0] + self.sample_size[1]):
                    indexs.append(self.data1[num1 % len(self.data1)])
                    num1 += 1

        return iter(indexs)

    def __len__(self) -> int:
        return self.total_epoch * self._batch_size

    def computer_epoch(self):
        if self.with_unlabel:
            if self.round_up:
                self.total_epoch = max(len(self.data1) // self.sample_size[0], len(self.data2) // self.sample_size[1])
            else:
                self.total_epoch = min(len(self.data1) // self.sample_size[0], len(self.data2) // self.sample_size[1])
        else:
            self.total_epoch = len(self.data1) // sum(self.sample_size)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch: int) -> None:
        self._batch_size = batch
        self.sample_size = [int(self.sample_ratio[0] * batch), batch - int(self.sample_ratio[0] * batch)]

    def set_seed(self, seed: int) -> None:
        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    @property
    def with_unlabel(self) -> bool:
        return self._with_unlabel

    @with_unlabel.setter
    def with_unlabel(self, unlabel: bool) -> None:
        self._with_unlabel = unlabel
        self.computer_epoch()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


if __name__ == "__main__":

    class TestSet(Dataset):
        def __init__(self, start=0, end=100) -> None:
            super().__init__()
            self.start = start
            self.end = end
            self.data = [i for i in range(start, end)]

        def __getitem__(self, index) -> Any:
            return self.data[index]

        def __len__(self):
            return len(self.data)

    datasets = ConcatDataset([TestSet(), TestSet(start=100, end=200)])
    batch = 8
    sampler = SemiSampler(datasets, batch, [2, 6])
    data_loader = DataLoader(dataset=datasets, batch_size=batch, sampler=sampler, num_workers=1)
    for idx, data in enumerate(data_loader):
        print(idx, data)
    data_loader.sampler.with_unlabel = True
    for idx, data in enumerate(data_loader):
        print(idx, data)
