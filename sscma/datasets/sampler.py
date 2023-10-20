from typing import Iterator, List, Optional, Any
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
        sample_ratio: List[int, float],
        shuffle: bool = True,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        assert len(sample_ratio) == len(
            dataset.cumulative_sizes
        ), "Sampling rate length must be equal to the number of datasets."

        super(SemiSampler, self).__init__(dataset, shuffle=shuffle, seed=seed, round_up=round_up)
        if seed is not None:
            self.set_seed(seed)

        self.dataset = dataset
        self.batch_size = batch_size
        sample_ratio = [scale / sum(sample_ratio) for scale in sample_ratio]
        self.sample_ratio = sample_ratio
        self.sample_size = [int(sample_ratio[0] * batch_size), batch_size - int(sample_ratio[0] * batch_size)]

        data_index = np.arange(dataset.cumulative_sizes[1])
        self.data1 = data_index[: dataset.cumulative_sizes[0]]
        self.data2 = data_index[dataset.cumulative_sizes[0] :]
        if shuffle:
            np.random.shuffle(self.data1)
            np.random.shuffle(self.data2)

        if round_up:
            self.epoch = max(len(self.data1) // self.sample_size[0], len(self.data2) // self.sample_size[1])
        else:
            self.epoch = min(len(self.data1) // self.sample_size[0], len(self.data2) // self.sample_size[1])

        self.datasets_len = dataset.cumulative_sizes[-1]

    def __iter__(self) -> Iterator[int]:
        indexs = []
        num1 = 0
        num2 = 0
        for i in range(self.epoch):
            for _ in range(self.sample_size[0]):
                indexs.append(self.data1[num1 % len(self.data1)])
                num1 += 1
            for _ in range(self.sample_size[1]):
                indexs.append(self.data2[num2 % len(self.data2)])
                num2 += 1

        return iter(indexs)

    def __len__(self) -> int:
        return self.epoch * self.batch_size

    def set_seed(self, seed: int) -> None:
        # set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)


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
