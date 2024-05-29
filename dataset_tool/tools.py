from typing import Any
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import serial
from tqdm import tqdm
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torchaudio.transforms as T
import random


def paa(time_series, segments):
    """
    Perform Piecewise Aggregate Approximation (PAA) on a given time series.
    
    Parameters:
    - time_series: numpy array, the original time series data
    - segments: int, the number of segments to divide the time series into
    
    Returns:
    - paa_series: numpy array, the PAA transformed time series
    """
    n = len(time_series)
    if n % segments == 0:
        segment_size = n // segments
        paa_series = np.mean(time_series.reshape(segments, segment_size), axis=1)
    else:
        paa_series = np.zeros(segments)
        for i in range(segments):
            start = i * n // segments
            end = (i + 1) * n // segments
            paa_series[i] = np.mean(time_series[start:end])

    return paa_series


def min_max_scale(X, max_val=1, min_val=0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max_val - min_val) + min_val
    return X_scaled


def generate_gaf(X, gaf_type='summation'):
    X_scaled = min_max_scale(X)  # 将时间序列数据缩放到 [0, 1]
    phi = np.arccos(X_scaled)  # 计算角度
    if gaf_type == 'summation':
        gaf = np.cos(phi[:, None] + phi[None, :])
    elif gaf_type == 'difference':
        gaf = np.sin(phi[:, None] - phi[None, :])
    else:
        raise ValueError("gaf_type must be 'summation' or 'difference'")
    return gaf


def add_noise(waveform, noise_factor=0.005):
    noise = torch.randn(waveform.size())
    augmented_waveform = waveform + noise_factor * noise
    return augmented_waveform


def time_shift(waveform):
    shift_factor = random.random() * 0.5
    shift = int(shift_factor * waveform.size(0))
    augmented_waveform = torch.roll(waveform, shifts=shift)
    return augmented_waveform


def pitch_shift(waveform, sample_rate, n_steps):
    transform = T.PitchShift(sample_rate, n_steps)
    augmented_waveform = transform(waveform)
    return augmented_waveform


def frequency_masking(mel_spectrogram, freq_mask_param=10):
    transform = T.FrequencyMasking(freq_mask_param=freq_mask_param)
    augmented_mel_spectrogram = transform(mel_spectrogram)
    return augmented_mel_spectrogram


class Signal_dataset(Dataset):

    def __init__(self, data_root, tag, data_len=60, transform=None):
        self.data_root = data_root
        self.tag = tag
        if tag == "Dynamic_Train":
            self.data_len = data_len
            self.ser = serial.Serial('COM3', 115200)
            if self.ser.isOpen():  # 判断串口是否成功打开
                print("打开串口成功。")
                print(self.ser.name)  # 输出串口号
            else:
                print("打开串口失败。")

        self.raw_sample = self.get_all_sample()
        self.sample = self.sample_process()

    def sample_process(self):
        if self.tag == "Dynamic_Train":
            sample = []
            for index, data in enumerate(self.raw_sample):
                data = self.raw_sample[index].decode().strip('\r\n').split(' ')[:-1]
                data = np.reshape(np.array(data).astype('float32'), (-1, 3))
                temp = []
                for i in range(3):
                    data_temp = paa(data[:, i], 64)
                    gadf_i = generate_gaf(data_temp, gaf_type='difference')
                    temp.append(gadf_i)
                gadf = np.array(temp)
                sample.append(gadf)
            return sample
        else:
            return self.raw_sample

    def get_all_sample(self):
        if self.tag == "Dynamic_Train":
            sample = []
            self.ser.reset_input_buffer()
            with tqdm(total=self.data_len) as pbar:
                for index, i in enumerate(self.ser):
                    if index == 0:
                        continue
                    sample.append(i)
                    pbar.update(1)
                    if index == self.data_len:
                        self.ser.close()
                        break

            sample = np.array(sample)
        else:
            sample = glob.glob(os.path.join(self.data_root, self.tag, "*.npy"), recursive=True)
        return sample

    def __getitem__(self, index: Any) -> Any:
        if self.tag == "Dynamic_Train":
            data = self.sample[index]
        else:
            npy_path = self.sample[index]
            data = np.load(npy_path)

        return data

    def __len__(self):
        return len(self.sample)


class Microphone_dataset(Dataset):

    def __init__(self, data_root, tag, sample_rate=16000, n_mels=32, data_len=10, transform=None):
        self.data_root = data_root
        self.tag = tag
        self.sample_rate = sample_rate
        self.mel_transform = MelSpectrogram(sample_rate, n_fft=256, n_mels=n_mels)
        self.db_transform = AmplitudeToDB()
        if tag == "Dynamic_Train":
            self.data_len = data_len
            self.ser = serial.Serial('COM5', 115200)
            if self.ser.isOpen():  # 判断串口是否成功打开
                print("打开串口成功。")
                print(self.ser.name)  # 输出串口号
            else:
                print("打开串口失败。")
        self.sample = self.get_all_sample()

    def get_all_sample(self):
        if self.tag == "Dynamic_Train":
            sample = []
            self.ser.reset_input_buffer()
            with tqdm(total=self.data_len) as pbar:
                for index, i in enumerate(self.ser):
                    if index == 0:
                        continue
                    data = i.decode().strip('\r\n').split(' ')[:-1]
                    data = np.reshape(np.array(data).astype('float32'), (-1, ))
                    sample.append(data)
                    pbar.update(1)
                    if index == self.data_len:
                        self.ser.close()
                        break

            sample = np.array(sample)
        else:
            sample = glob.glob(os.path.join(self.data_root, self.tag, "*.npy"), recursive=True)
        return sample

    def __getitem__(self, index: Any) -> Any:
        # 适用于音频数据的代码还未写，具体实现方法参考信号
        if self.tag == "Dynamic_Train":
            data = self.sample[index]
            data = data
            data = data.astype(np.float32)
            data = torch.tensor(data)
            data = data
            data = self.augment_waveform(data)
            mel_spectrogram = self.mel_transform(data)
            mel_spectrogram_db = self.db_transform(mel_spectrogram) / 100
            mel_spectrogram_db = mel_spectrogram_db[:, :-1]
            data = mel_spectrogram_db.unsqueeze(0)
        else:
            npy_path = self.sample[index]
            data = np.load(npy_path)

        return data

    def augment_waveform(self, waveform):
        waveform = add_noise(waveform)
        return waveform

    def __len__(self):
        return len(self.sample)
