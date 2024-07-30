from torch.utils.data import Sampler
import numpy as np
import math


class LanceDistributedSampler(Sampler):
    """Distributed sampler for LanceDataset.

    Param:
        total_data: The total length of the sampled data set
        rank: The current sampler sequence number
        total_worker: The total number of samplers
        shuffle: Whether to disrupt the sampled index order
    """

    def __init__(self, total_data, rank, total_worker, shuffle=True):

        self.rank = rank
        self.total_worker = total_worker
        self.total_data = total_data
        self.epoch = 0

        self.step_len = math.ceil(self.total_data / self.total_worker)

        start_idx = self.rank * self.step_len
        end_idx = min((self.rank + 1) * self.step_len, self.total_data)

        self.data_indices = list(range(start_idx, end_idx))

        if shuffle:
            np.random.shuffle(self.data_indices)

    def __iter__(self):
        yield from self.data_indices

    def __len__(self):
        return len(self.data_indices)
