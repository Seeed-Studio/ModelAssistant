import numpy as np
import matplotlib.pyplot as plt
import serial
import cv2
import pywt
from scipy.signal import stft
from scipy.signal import firwin, lfilter


def preprocess(data):
    data = data * 0.05
    data = np.floor(data)
    data = data * 20
    return data.astype('float32')


def long_time_fourier_transform(signal, fs, nperseg=32, noverlap=None):

    # nperseg = 256  # 每段的长度
    # nfft = 2048
    # min_val = np.min(signal)
    # max_val = np.max(signal)
    # signal = (signal - min_val) / (max_val - min_val)
    f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap, nfft=128)
    Zxx = np.real(Zxx)
    min_val = np.min(Zxx)
    max_val = np.max(Zxx)
    Zxx = (Zxx - min_val) / (max_val - min_val)
    # Zxx = min_max_scale(Zxx[:-1, :-1])
    return Zxx


def CWT(signal, scales=np.arange(1, 65), wavelet='morl'):
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
    return np.abs(coefficients)


def markov_transition_field(time_series, n_bins=16):
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
    return gaf


# 示例用法
if __name__ == "__main__":
    # 生成示例时间序列数据
    ser = serial.Serial('COM8', 115200)
    data = []
    data_res = np.zeros((3, 65, 65))
    res_tag = 1
    for index, i in enumerate(ser):
        if (index + 1) % 2048 != 0:
            temp = i.decode().strip('\r\n').split(' ')
            data = data + temp
            continue
        temp = i.decode().strip('\r\n').split(' ')
        data = data + temp
        data = np.reshape(np.array(data).astype('float32'), (-1, 3))
        data = preprocess(data)
        # break
        # data = ser.readline().decode()
        # 打印读取到的数据
        print("Received:", data)
        t = np.linspace(0, 2 * np.pi, 100)
        X = data
        gadf = []
        # 计算 GASF 和 GADF
        for i in range(3):
            # gasf = generate_gaf(X, gaf_type='summation'
            data_temp = X[:, i]
            nyquist_rate = 6667 / 2.0
            cutoff_freq = 20.0  # 截止频率
            numtaps = 100  # 滤波器系数数量，越大则滤波器越陡峭
            fir_coeff = firwin(numtaps, cutoff_freq / nyquist_rate)
            data_temp = lfilter(fir_coeff, 1.0, data_temp)
            data_temp = paa(data_temp, 1024)
            # gadf_i = generate_gaf(data_temp, gaf_type='summation')
            # gadf_i = markov_transition_field(data_temp)
            gadf_i = long_time_fourier_transform(data_temp, fs=6667)
            # gadf_i = CWT(data_temp)
            # gadf_i = cv2.resize(gadf_i, (128,128), interpolation=cv2.INTER_LINEAR)
            gadf.append(gadf_i)
        gadf = np.array(gadf)
        # gadf[0:10] = 0
        # gadf = gadf[:, 32:96, 32:96]
        # gadf_diff = (gadf - data_res) / 2
        data_res = gadf
        gadf_diff = np.transpose(gadf, (1, 2, 0))
        image = (gadf_diff * 255).astype(np.uint8)
        # gadf = np.transpose(gadf, (1, 2, 0))
        # image = (gadf * 255).astype(np.uint8)
        # 绘制结果
        cv2.imshow('55', image)
        cv2.waitKey(1)
        data = []
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.imshow(gasf, origin='lower')
    # plt.title('Gramian Angular Summation Field (GASF)')
    # plt.colorbar()

    # plt.subplot(1, 2, 2)
    # plt.imshow(gadf, cmap='rainbow', origin='lower')
    # plt.title('Gramian Angular Difference Field (GADF)')
    # plt.colorbar()

    # plt.tight_layout()
    # plt.show()
