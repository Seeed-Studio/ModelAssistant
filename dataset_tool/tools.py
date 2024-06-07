from typing import Any
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

sample_rate = 8192


def sample_entropy(U, m, r):
    """
    计算样本熵 (SampEn)
    
    参数：
    U: 输入时间序列
    m: 嵌入维数
    r: 公差（tolerance），通常取时间序列标准差的0.2倍
    
    返回：
    SampEn值
    """
    N = len(U)

    def _phi(m):
        X = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) / (N - m + 1)
        return np.sum(C - 1) / (N - m)

    return -np.log(_phi(m + 1) / _phi(m))


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


def long_time_fourier_transform(signal, fs, nperseg=32, noverlap=None, nfft=128):

    # nperseg = 256  # 每段的长度
    # nfft = 2048
    f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Zxx = np.real(Zxx)
    min_val = np.min(Zxx)
    max_val = np.max(Zxx)
    Zxx = (Zxx - min_val) / (max_val - min_val)
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


def sample_c_process(raw_sample):
    data = raw_sample
    data = np.reshape(np.array(data).astype('float32'), (-1, 3))
    data = preprocess(data)
    gadf = []
    c = []
    for i in range(3):
        data_temp = data[:, i]
        nyquist_rate = 6667 / 2.0
        cutoff_freq = 2.0  # 截止频率
        numtaps = 200  # 滤波器系数数量，越大则滤波器越陡峭
        fir_coeff = firwin(numtaps, cutoff_freq / nyquist_rate)
        data_temp = lfilter(fir_coeff, 1.0, data_temp)
        data_c = paa(data_temp, 1024)
        data_temp = paa(data_temp, 32)
        gadf_i = markov_transition_field(data_temp)
        c_i = long_time_fourier_transform(data_c, fs=6667, nperseg=64, nfft=64)
        gadf.append(gadf_i)
        c.append(c_i)

    c = np.array(c)
    gadf = np.array(gadf)
    label = gadf

    return gadf.astype('float32'), c.astype('float32'), label.astype('float32')


class Signal_dataset(Dataset):

    def __init__(self, data_root, tag, data_len=200, transform=None):
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
        # self.sample = self.sample_process()
        # self.sample = self.sample_diff_process()
        self.sample = self.sample_c_process()

    def sample_process(self):
        if self.tag == "Dynamic_Train":
            sample = []
            for index, data in enumerate(self.raw_sample):
                data = self.raw_sample[index]
                data = np.reshape(np.array(data).astype('float32'), (-1, 3))
                data = preprocess(data)
                temp = []
                for i in range(3):
                    data_temp = data[:, i]
                    nyquist_rate = 6667 / 2.0
                    cutoff_freq = 1.0  # 截止频率
                    numtaps = 100  # 滤波器系数数量，越大则滤波器越陡峭
                    fir_coeff = firwin(numtaps, cutoff_freq / nyquist_rate)
                    data_temp = lfilter(fir_coeff, 1.0, data_temp)
                    data_temp = paa(data_temp, 1024)
                    # gadf_i = generate_gaf(data_temp, gaf_type='summation')
                    # gadf_i = markov_transition_field(data_temp)
                    gadf_i = long_time_fourier_transform(data_temp, fs=6667)
                    # temp.append(gadf_i)
                    temp.append(gadf_i)
                gadf = np.array(temp).astype('float32')
                sample.append(gadf)
            return sample
        else:
            return self.raw_sample

    def sample_diff_process(self):
        if self.tag == "Dynamic_Train":
            sample = []
            data_res = np.zeros((3, 64, 64))
            for index, data in enumerate(self.raw_sample):
                data = self.raw_sample[index]
                data = np.reshape(np.array(data).astype('float32'), (-1, 3))
                data = preprocess(data)
                temp = []
                Signal_Description = []
                for i in range(3):
                    data_temp = data[:, i]
                    nyquist_rate = 6667 / 2.0
                    cutoff_freq = 1.0  # 截止频率
                    numtaps = 200  # 滤波器系数数量，越大则滤波器越陡峭
                    fir_coeff = firwin(numtaps, cutoff_freq / nyquist_rate)
                    data_temp = lfilter(fir_coeff, 1.0, data_temp)
                    data_temp = paa(data_temp, 64)
                    Signal_Description.append(calculate_rms(data_temp) / 1024)
                    # gadf_i = generate_gaf(data_temp, gaf_type='difference')
                    gadf_i = markov_transition_field(data_temp)
                    # gadf_i = long_time_fourier_transform(data_temp, fs=6667)
                    # temp.append(gadf_i)
                    temp.append(gadf_i)
                gadf = np.array(temp)
                label = np.array(temp)
                Signal_Description = np.array(Signal_Description)
                gadf = gadf + np.random.randn(3, 64, 64) * 0
                # gadf[:, 0:15, :] = 0
                # gadf = gadf[:, 32:96, 32:96]
                # gadf_diff = (gadf - data_res) / 2
                # data_res_return = data_res.copy()
                sample.append((data_res.astype('float32'), gadf.astype('float32'), label.astype('float32'), Signal_Description.astype('float32')))
                data_res = gadf
            return sample
        else:
            return self.raw_sample

    def sample_c_process(self):
        if self.tag == "Dynamic_Train":
            sample = []
            for index, data in enumerate(self.raw_sample):
                gadf, c, label = sample_c_process(data)
                sample.append((gadf, c, label))
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
        # waveform = time_shift(waveform)
        # waveform = time_stretch(waveform, self.sample_rate)
        # waveform = pitch_shift(waveform, self.sample_rate, n_steps=2)
        return waveform

    def __len__(self):
        return len(self.sample)
