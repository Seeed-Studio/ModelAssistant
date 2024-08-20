import time
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import serial
import cv2
import numpy as np
from torch.multiprocessing import Process, Pipe
from dataset_tool.tools import Signal_dataset
from misc import tools
from matplotlib import pyplot as plt
import dataset_tool.tools as dt


def write_Signal_to_tensor(pipe):
    # 实时读取信号数据
    produce, consume = pipe
    consume.close()
    ser = serial.Serial('COM9', 115200)
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
        print(ser.name)  # 输出串口号
    else:
        print("打开串口失败。")
    while True:
        data = []
        # a = time.time()
        for index, i in enumerate(ser):
            # b = time.time()v
            # print(b - a)
            if (index + 1) % dt.sample_rate != 0:  # 根据设定好的采样率组装数据点，足够组成一条数据后便发送給消费者
                temp = i.decode().strip('\r\n').split(' ')[:-1]
                data = data + temp
                continue
            temp = i.decode().strip('\r\n').split(' ')[:-1]
            data = data + temp
            # b = time.time()
            # print(b - a)
            # a = time.time()

            produce.send(data)  # 将写入的信息放入管道
            data = []
            ser.flushInput()  # 清楚串口缓存
            # count = ser.inWaiting()
            # print(count)
            # b = time.time()
            # print(b - a)
            # a = time.time()


def disp_preprocess_data(pipe):
    # TODO 需要兼容加载方式，音频数据与信号数据的加载方式是不同的，目前仅以取消注释的方法来改变
    # mel_transform = MelSpectrogram(96000, n_fft=32, n_mels=16)
    # db_transform = AmplitudeToDB()
    produce, consume = pipe
    produce.close()
    data_mean = np.load('x_data_mean.npy')
    data_std = np.load('x_data_std.npy')
    # data_res = np.zeros((3, 64, 64))
    while True:
        if consume.poll():
            data = consume.recv()  # 从管道中获取写入的信息
            data, c, label = dt.sample_c_process(data, data_mean, data_std)
            data = np.transpose(data, (1, 2, 0))
            data = cv2.resize(data, (256, 256), interpolation=cv2.INTER_LINEAR)
            image = (data * 255).astype(np.uint8)
            # 绘制结果
            cv2.imshow('55', image)
            cv2.waitKey(1)
            time.sleep(0.1)


def work_pipe():
    pipe = Pipe()

    # producer_process = Process(target=write_Sound_to_tensor, args=(pipe, ))
    producer_process = Process(target=write_Signal_to_tensor, args=(pipe, ))  # 调用生产者
    consumer_process = Process(target=disp_preprocess_data, args=(pipe, ))  # 调用消费者

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()


if __name__ == '__main__':
    work_pipe()
