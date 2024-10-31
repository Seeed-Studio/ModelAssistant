# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any
import glob

import torch
import numpy as np
from torch.utils.data import Dataset
import torchaudio.transforms as T


def generate_Mel_DBtans(sample_rate=96000, n_fft=64, n_mels=32):
    mel_transform = T.MelSpectrogram(sample_rate, n_fft=n_fft, n_mels=n_mels)
    db_transform = T.AmplitudeToDB()
    return mel_transform, db_transform


class Signal_dataset(Dataset):

    def __init__(
        self,
        data_root=None,
    ):
        self.data_root = data_root
        self.sample = self.get_all_sample()

    def get_all_sample(self):
        sample = glob.glob(osp.join(self.data_root, self.tag, "*.npy"), recursive=True)
        return sample

    def __getitem__(self, index: Any) -> Any:
        npy_path = self.sample[index]
        data = np.load(npy_path)
        data = (data[0], data[1], data[2])
        return torch.from_numpy(data)

    def __len__(self):
        return len(self.sample)


class Microphone_dataset(Dataset):
    def __init__(
        self,
        data_root=None,
    ):
        self.data_root = data_root
        self.mel_transform, self.db_transform = generate_Mel_DBtans()

        self.sample = self.get_all_sample()

    def get_all_sample(self):
        sample = glob.glob(osp.join(self.data_root, "*.npy"), recursive=True)
        return sample

    def __getitem__(self, index: Any) -> Any:
        npy_path = self.sample[index]
        data = np.load(npy_path)
        return torch.from_numpy(data)

    def __len__(self):
        return len(self.sample)
