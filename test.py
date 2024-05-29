import numpy as np
import matplotlib.pyplot as plt
import serial
import cv2


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
    ser = serial.Serial('COM3', 115200)
    for index, i in enumerate(ser):
        if index == 0:
            continue
        data = i.decode().strip('\r\n').split(' ')[:-1]
        data = np.reshape(np.array(data).astype('float32'), (-1, 3))
        break
    # data = ser.readline().decode()

    # 打印读取到的数据
    print("Received:", data)
    t = np.linspace(0, 2 * np.pi, 100)
    X = data
    gadf = []
    # 计算 GASF 和 GADF
    for i in range(3):
        # gasf = generate_gaf(X, gaf_type='summation'
        data_temp = paa(X[:, i], 128)
        gadf_i = generate_gaf(data_temp, gaf_type='difference')
        # gadf_i = cv2.resize(gadf_i, (128,128), interpolation=cv2.INTER_LINEAR)
        gadf.append(gadf_i)
    gadf = np.array(gadf)
    gadf = np.transpose(gadf, (1, 2, 0))
    image = (gadf * 255).astype(np.uint8)
    # 绘制结果
    cv2.imshow('55', image)
    cv2.waitKey(0)
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
