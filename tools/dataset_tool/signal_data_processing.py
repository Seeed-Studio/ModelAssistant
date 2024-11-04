import numpy as np
import pandas as pd
import os
import argparse
import tools as dt

import cv2
import pywt

from scipy.signal import firwin, lfilter, stft


def CWT(signal, scales=np.arange(1, 32), wavelet="cgau8"):
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, method="conv")
    return coefficients


def global_mean_std_estimate(dataset, data_type="Signal"):
    mean = 0
    std = 0
    data_len = 0
    for data in dataset:
        if data_type == "Signal":
            data = np.reshape(np.array(data).astype("float32"), (-1, 3))
        elif data_type == "Micro":
            data = np.reshape(np.array(data).astype("float32"), (-1,))
        mean = mean + data.mean()
        data_len = data_len + 1
    mean = mean / data_len
    for data in dataset:
        if data_type == "Signal":
            data = np.reshape(np.array(data).astype("float32"), (-1, 3))
        elif data_type == "Micro":
            data = np.reshape(np.array(data).astype("float32"), (-1,))
        temp = (data - mean) * (data - mean)
        temp = temp.mean()
        std = std + temp
    std = std / data_len
    std = np.sqrt(std)
    np.save("x_data_mean", mean)
    np.save("x_data_std", std)
    return mean, std


def mean_std_scale(x, mean, std):
    for i in range(len(x)):
        x[i] = (x[i] - mean) / std
    return x


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


def markov_transition_field(time_series, n_bins=64):
    """
    Calculate the Markov Transition Field (MTF) of a given time series.

    Parameters:
    - time_series: numpy array, the original time series data
    - n_bins: int, the number of bins to divide the time series into

    Returns:
    - mtf: numpy array, the MTF of the time series
    """
    min_val = np.min(time_series)
    max_val = np.max(time_series)
    norm_time_series = (time_series - min_val) / (max_val - min_val)

    bins = np.linspace(0, 1, n_bins)
    digitized = np.digitize(norm_time_series, bins)
    digitized[digitized == n_bins] = n_bins - 1

    transition_matrix = np.zeros((n_bins, n_bins))
    for i, j in zip(digitized[:-1], digitized[1:]):
        transition_matrix[i - 1, j - 1] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_matrix / row_sums

    n = len(digitized)
    mtf = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mtf[i, j] = transition_matrix[digitized[i] - 1, digitized[j] - 1]

    return mtf.astype("float32")


def preprocess(data):
    factor = 5.12
    data = data * factor
    data = np.floor(data)
    data = data / factor
    return data.astype("float32")


def sample_c_process(raw_sample, mean=None, std=None, sample_rate=115200):
    data = raw_sample
    data = np.reshape(np.array(data).astype("float32"), (-1, 3))
    data = preprocess(data)
    if mean is not None:
        data = mean_std_scale(data, mean, std)
    x = []
    c = []
    use_fir = False

    for i in range(3):
        data_temp = data[:, i]
        nyquist_rate = sample_rate / 2.0
        cutoff_freq = 90.0

        numtaps = 200
        fir_coeff = firwin(numtaps, cutoff_freq / nyquist_rate)
        data_temp = lfilter(fir_coeff, 1.0, data_temp)
        data_temp_c = paa(data_temp, 32)
        if use_fir:
            data_temp_x = data_temp
        else:
            data_temp_x = data[:, i]

        c_i = markov_transition_field(data_temp_c)
        data_temp = cv2.resize(data_temp, (1, 32), interpolation=cv2.INTER_LINEAR)

        data_temp = np.squeeze(data_temp)
        x_i = CWT(data_temp_x, scales=np.arange(10, 10 + 16), wavelet="morl")

        x_i = cv2.resize(x_i, (32, 32), interpolation=cv2.INTER_LINEAR)
        x.append(x_i)
        c.append(c_i)

    c = np.array(c)
    x = np.array(x)
    label = x

    return x.astype("float32"), c.astype("float32"), label.astype("float32")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", "-t", type=str, default="train")
    parser.add_argument("--file_path", "-f", type=str, default="serial_data.csv")
    parser.add_argument("--sample_rate", "-sr", type=int, default=115200)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    file_path, tag, sample_rate = args.file_path, args.tag, args.sample_rate
    datas = pd.read_csv(file_path)
    process_data = []

    mean, std = global_mean_std_estimate(datas.values)
    for iter, item in datas.iterrows():
        data, c, label = sample_c_process(item, mean=mean, std=std,sample_rate=sample_rate)
        process_data.append([data, c, label])

    process_data = np.array(process_data)

    save_and_split_data(process_data.astype(np.float32), tag, file_path)


def save_and_split_data(data, tag, data_path):
    os.makedirs("datasets", exist_ok=True)
    save_path = os.path.join("datasets", tag)
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(data_path))[0]

    for index, item in enumerate(data):
        file_path = os.path.join(save_path, f"{index:04d}_{file_name}")
        np.save(file_path, item)


if __name__ == "__main__":
    main()
