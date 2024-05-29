'''
该文件用于处理采集好的三轴震动信号
'''

import pandas as pd
import numpy as np
import os
from tools import paa, generate_gaf
import sys

sys.path.append("../")
# from dataset_tool.tools import paa, generate_gaf


def item_process(item, data_len):
    data = np.reshape(item, (-1, 3))
    temp = []
    for i in range(3):
        data_temp = paa(data[:, i], data_len)
        gadf_i = generate_gaf(data_temp, gaf_type='difference')
        temp.append(gadf_i)
    gadf = np.array(temp)
    return gadf


def data_process(data_path, data_len):
    data = pd.read_csv(data_path)
    process_data = []
    for iter, item in data.iterrows():
        item = item_process(item, data_len)
        process_data.append(item)
    process_data = np.array(process_data)
    return process_data


def save_and_split_data(data, tag, data_path):
    os.makedirs("datasets", exist_ok=True)
    save_path = os.path.join("datasets", tag)
    os.makedirs(save_path, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(data_path))[0]

    for index, item in enumerate(data):
        file_path = os.path.join(save_path, f"{index:04d}_{file_name}")

        np.save(file_path, item)


if __name__ == '__main__':
    data_path = "serial_data.csv"
    tag = "Train"
    data_len = 64  # 用于指定时间序列聚合后的数据长度
    data = data_process(data_path, data_len).astype('float32')
    save_and_split_data(data, tag, data_path)
