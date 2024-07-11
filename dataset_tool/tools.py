import math
from typing import Any
import pywt
import scipy
from torch.utils.data import Dataset
# from einops import rearrange
import numpy as np
import os
import glob
import serial
from tqdm import tqdm
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torchaudio.transforms as T
import random
import time
from scipy.signal import stft
from scipy.signal import firwin, lfilter
import cv2
from matplotlib import pyplot as plt

sample_rate = 4096


def haar_wavelet_transform(signal):
    n = len(signal)
    output = np.zeros_like(signal)
    while n > 1:
        n //= 2
        output[:n] = (signal[::2] + signal[1::2]) / np.sqrt(2)
        output[n:2 * n] = (signal[::2] - signal[1::2]) / np.sqrt(2)
        signal = output[:2 * n]
    return output


def inverse_haar_wavelet_transform(coeffs):
    n = 1
    output = np.zeros_like(coeffs)
    output[:n] = coeffs[:n]
    while n * 2 <= len(coeffs):
        n *= 2
        temp = output[:n].copy()
        output[0:n:2] = (temp[:n // 2] + coeffs[n // 2:n]) / np.sqrt(2)
        output[1:n:2] = (temp[:n // 2] - coeffs[n // 2:n]) / np.sqrt(2)
    return output


def generate_spectrogram(signal, scales, wavelet_transform):
    n = len(signal)
    spectrogram = np.zeros((len(scales), n))
    for i, scale in enumerate(scales):
        scaled_signal = signal[::scale]
        transformed_signal = wavelet_transform(scaled_signal)
        spectrogram[i, :len(transformed_signal)] = np.abs(transformed_signal)
    return spectrogram


def calculate_rms(signal):
    """
    计算一维时序信号的均方根值（RMS）

    参数：
    signal (array-like): 输入的时序信号

    返回：
    float: 信号的均方根值（RMS）
    """
    signal = np.array(signal)
    rms = np.sqrt(np.mean(signal**2))
    return rms


def preprocess(data):
    data = data * 0.021
    data = np.floor(data)
    data = data * 50
    return data


def CWT(signal, scales=np.arange(1, 32), wavelet='cgau8'):
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    # Zxx = np.abs(coefficients)
    # min_val = np.min(Zxx)
    # max_val = np.max(Zxx)
    # Zxx = (Zxx - min_val) / (max_val - min_val)
    return coefficients


def long_time_fourier_transform(signal, fs, nperseg=32, noverlap=None, nfft=128):

    # nperseg = 256  # 每段的长度
    # nfft = 2048
    f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Zxx = np.real(Zxx)
    # min_val = np.min(Zxx)
    # max_val = np.max(Zxx)
    # Zxx = (Zxx - min_val) / (max_val - min_val)
    # Zxx = min_max_scale(Zxx[:-1, :-1]).astype('float32')
    return Zxx[:-1, :-1]


def markov_transition_field(time_series, n_bins=64):
    """
    计算时间序列的马尔可夫迁移场（MTF）
    
    参数:
    time_series (array-like): 一维时间序列信号
    n_bins (int): 将时间序列分箱的数量
    
    返回:
    mtf (2D array): 马尔可夫迁移场矩阵
    """
    # 将时间序列归一化到 [0, 1] 范围内
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    norm_time_series = (time_series - min_val) / (max_val - min_val)
    # 将归一化后的时间序列分箱
    bins = np.linspace(0, 1, n_bins)
    digitized = np.digitize(norm_time_series, bins)
    digitized[digitized == n_bins] = n_bins - 1  # 修正索引边界问题

    # 计算转移矩阵
    transition_matrix = np.zeros((n_bins, n_bins))
    for (i, j) in zip(digitized[:-1], digitized[1:]):
        transition_matrix[i - 1, j - 1] += 1

    # 将转移矩阵归一化为概率
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 防止除以零
    transition_matrix = transition_matrix / row_sums

    # 计算马尔可夫迁移场（MTF）
    n = len(digitized)
    mtf = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mtf[i, j] = transition_matrix[digitized[i] - 1, digitized[j] - 1]

    return mtf.astype('float32')


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


def mean_std_scale(x, mean, std):
    x = ((x - mean) / (std + 1e-5)) / 100
    return x


def generate_gaf(X, gaf_type='summation'):
    X_scaled = min_max_scale(X)  # 将时间序列数据缩放到 [0, 1]
    phi = np.arccos(X_scaled)  # 计算角度
    if gaf_type == 'summation':
        gaf = np.cos(phi[:, None] + phi[None, :])
    elif gaf_type == 'difference':
        gaf = np.sin(phi[:, None] - phi[None, :])
    else:
        raise ValueError("gaf_type must be 'summation' or 'difference'")
    return gaf.astype('float32')


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


def sample_c_process(
    raw_sample,
    mean=None,
    std=None,
):
    data = raw_sample
    data = np.reshape(np.array(data).astype('float32'), (-1, 3))
    data = preprocess(data)
    x = []
    c = []
    for i in range(3):
        data_temp = data[:, i]
        nyquist_rate = 6667 / 2.0
        cutoff_freq = 160.0  # 截止频率
        numtaps = 200  # 滤波器系数数量，越大则滤波器越陡峭
        fir_coeff = firwin(numtaps, cutoff_freq / nyquist_rate)
        data_temp = lfilter(fir_coeff, 1.0, data_temp)
        # data_c = paa(data_temp, 1024)
        data_temp_m = paa(data_temp, 32)
        c_i = markov_transition_field(data_temp_m)
        # x_i = long_time_fourier_transform(data_c, fs=6667, nperseg=64, nfft=64)
        x_i = CWT(data_temp, np.arange(9, 41), wavelet='morl')
        x_i = cv2.resize(x_i, (32, 32), interpolation=cv2.INTER_LINEAR)
        x.append(x_i)
        c.append(c_i)

    c = np.array(c)
    x = np.array(x)
    if mean is not None:
        x = mean_std_scale(x, mean, std)
    label = x

    return x.astype('float32'), c.astype('float32'), label.astype('float32')


def sample_mico_process(x, mel_transform, db_transform):
    x = np.reshape(np.array(x).astype('float32'), (-1, ))
    x = torch.tensor(x) / 1024
    mel_spectrogram = mel_transform(x)
    mel_spectrogram_db = db_transform(mel_spectrogram) / 100
    mel_spectrogram_db = mel_spectrogram_db[:, :-1]
    x = mel_spectrogram_db.unsqueeze(0)
    return x, x, x


class Signal_dataset(Dataset):

    def __init__(self, data_root, tag, data_len=100, transform=None):
        self.data_root = data_root
        self.tag = tag
        if tag == "Dynamic_Train":
            self.data_len = data_len
            self.ser = serial.Serial('COM8', 115200)
            if self.ser.isOpen():  # 判断串口是否成功打开
                print("打开串口成功。")
                print(self.ser.name)  # 输出串口号
            else:
                print("打开串口失败。")

        self.raw_sample = self.get_all_sample()
        self.sample = self.sample_c_process()

    def sample_c_process(self):
        if self.tag == "Dynamic_Train":
            sample = []
            std_record = []
            mean_record = []
            for index, data in enumerate(self.raw_sample):
                gadf, c, label = sample_c_process(data)
                mean_record.append([np.mean(gadf[0]), np.mean(gadf[1]), np.mean(gadf[2])])
                std_record.append([np.std(gadf[0]), np.std(gadf[1]), np.std(gadf[2])])
                sample.append([gadf, c, label])

            mean_record = np.array(mean_record)
            std_record = np.array(std_record)
            data_mean = np.sum(mean_record, axis=0, keepdims=True) / self.data_len
            data_mean = np.transpose(np.expand_dims(data_mean, axis=0), (2, 1, 0))
            data_std = np.sum(std_record, axis=0, keepdims=True) / self.data_len
            data_std = np.transpose(np.expand_dims(data_std, axis=0), (2, 1, 0))
            for index, data in enumerate(sample):
                sample[index][0] = ((data[0] - data_mean) / (data_std + 1e-5)) / 100
            np.save('x_data_mean', data_mean)
            np.save('x_data_std', data_std)
            return sample
        else:
            return self.raw_sample

    def get_all_sample(self):
        if self.tag == "Dynamic_Train":
            sample = []
            self.ser.reset_input_buffer()
            with tqdm(total=self.data_len) as pbar:
                data = []
                record = 0
                for index, i in enumerate(self.ser):
                    # 从串行端口读取数据
                    if (index + 1) % sample_rate != 0:
                        temp = i.decode().strip('\r\n').split(' ')
                        data = data + temp
                        continue
                    temp = i.decode().strip('\r\n').split(' ')
                    data = data + temp
                    sample.append(data)
                    pbar.update(1)
                    data = []
                    record = record + 1
                    # time.sleep(random.random() * 0.1)
                    if record == self.data_len:
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
            data = (data[0], data[1], data[2])

        return data

    def __len__(self):
        return len(self.sample)


class Microphone_dataset(Dataset):

    def __init__(self, data_root, tag, sample_rate=96000, n_mels=16, data_len=10, transform=None):
        self.data_root = data_root
        self.tag = tag
        self.sample_rate = sample_rate
        self.mel_transform = MelSpectrogram(sample_rate, n_fft=32, n_mels=n_mels)

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
        self.sample = self.sample_process()

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

    def sample_process(self):
        sample = []
        for data in self.sample:
            data = data.astype(np.float32) / 1024
            data = torch.tensor(data)
            data = self.augment_waveform(data)
            data, c, label = sample_mico_process(data, self.mel_transform, self.db_transform)
            sample.append([data, c, label])

        return sample

    def __getitem__(self, index: Any) -> Any:
        # 适用于音频数据的代码还未写，具体实现方法参考信号
        if self.tag == "Dynamic_Train":
            data = self.sample[index]
        else:
            npy_path = self.sample[index]
            data = np.load(npy_path)

        return data

    def augment_waveform(self, waveform):
        waveform = add_noise(waveform)
        return waveform

    def __len__(self):
        return len(self.sample)
