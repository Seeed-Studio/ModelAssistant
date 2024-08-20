'''
该文件用于处理采集好的三轴震动信号
'''
import numpy as np
import pandas as pd
import os
import tools as dt
import sys

sys.path.append("../")
# from dataset_tool.tools import paa, generate_gaf


def data_process(data_path):
    datas = pd.read_csv(data_path)
    process_data = []
    # std_record = []
    # mean_record = []
    mean, std = dt.global_mean_std_estimate(datas.values)
    for iter, item in datas.iterrows():
        data, c, label = dt.sample_c_process(item, mean=mean, std=std)
        # mean_record.append([np.mean(data[0]), np.mean(data[1]), np.mean(data[2])])
        # std_record.append([np.std(data[0]), np.std(data[1]), np.std(data[2])])
        process_data.append([data, c, label])

    # mean_record = np.array(mean_record)
    # std_record = np.array(std_record)
    # data_mean = np.sum(mean_record, axis=0, keepdims=True) / len(mean_record)
    # data_mean = np.transpose(np.expand_dims(data_mean, axis=0), (2, 1, 0))
    # data_std = np.sum(std_record, axis=0, keepdims=True) / len(std_record)
    # data_std = np.transpose(np.expand_dims(data_std, axis=0), (2, 1, 0))

    # for index, data in enumerate(process_data):
    #     process_data[index][0] = ((data[0] - data_mean) / (data_std + 1e-5)) / 10

    # np.save('x_data_mean', data_mean)
    # np.save('x_data_std', data_std)
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
