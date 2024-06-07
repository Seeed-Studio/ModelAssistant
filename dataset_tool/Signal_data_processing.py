'''
该文件用于处理采集好的三轴震动信号
'''

import pandas as pd
import numpy as np
import os
import tools as dt
import sys

sys.path.append("../")
# from dataset_tool.tools import paa, generate_gaf


def data_process(data_path):
    datas = pd.read_csv(data_path)
    process_data = []
    for iter, item in datas.iterrows():
        data, c, label = dt.sample_c_process(item)
        process_data.append((data, c, label))
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
    # data_path = "anomaly_rest.csv"
    data_path = "serial_data.csv"
    tag = "Train"
    data = data_process(data_path).astype('float32')
    save_and_split_data(data, tag, data_path)
