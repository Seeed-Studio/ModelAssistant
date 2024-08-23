from scipy import signal
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import serial
import cv2
import numpy as np
from torch.multiprocessing import Process, Pipe
from dataset_tool.tools import Signal_dataset
from misc import tools
from model import models
from matplotlib import pyplot as plt
import dataset_tool.tools as dt
from scipy.signal import firwin, lfilter


def diff_block(data):
    zero_array = np.zeros((1, 3)).astype('float32')
    data_diff = np.diff(data.T, n=1).T
    data_diff = np.vstack((data_diff, zero_array))
    data = np.hstack((data, data_diff))
    return data


def psnr(loss):
    psnr = 20 * np.log10(1 / np.sqrt(loss))
    return psnr


def Sound_process(data, mel_transform, db_transform):
    data = np.reshape(np.array(data).astype('float32'), (-1, ))
    data = torch.tensor(data)
    mel_spectrogram = mel_transform(data)
    mel_spectrogram_db = db_transform(mel_spectrogram) / 100
    return mel_spectrogram_db.unsqueeze(0).numpy()


def write_Sound_to_tensor(pipe):
    # 实时读取音频数据
    produce, consume = pipe
    consume.close()
    ser = serial.Serial('COM5', 115200)
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
        print(ser.name)  # 输出串口号
    else:
        print("打开串口失败。")
    while True:
        ser.flushInput()
        for index, i in enumerate(ser):
            if index == 0:
                continue
            data = i.decode().strip('\r\n').split(' ')[:-1]
            if len(data) != 1024:
                continue
            produce.send(data)  # 将写入的信息放入管道
            ser.flushInput()


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
            # b = time.time()
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


def read_from_tensor(pipe, model, loss):
    # TODO 需要兼容加载方式，音频数据与信号数据的加载方式是不同的，目前仅以取消注释的方法来改变
    produce, consume = pipe
    produce.close()
    data_mean = np.load('x_data_mean.npy')
    data_std = np.load('x_data_std.npy')
    score_list = []
    while True:
        if consume.poll():
            data = consume.recv()  # 从管道中获取写入的信息
            # data, c, label = dt.sample_c_process(data, data_mean, data_std)
            # 若要进行Micro的评估则取消下面两行注释
            mel_transform, db_transform = dt.generate_Mel_DBtans()
            data, c, label = dt.sample_micro_process(data, mel_transform, db_transform, data_mean, data_std)
            label = torch.tensor(label).unsqueeze(0)
            data_tensor = torch.tensor(data).unsqueeze(0)
            c = torch.tensor(c).unsqueeze(0)
            c_label = c
            input = torch.vstack((data_tensor, c))
            x, c = model(input)

            loss_cwt, loss_mar = tools.col_psnr(x, data_tensor, c, c_label)
            score_list.append(loss_cwt.item())
            plt.clf()
            plt.plot(score_list, 'r', label='Anomally Score')
            plt.title('Anomally Score')
            plt.legend()
            plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
            plt.ioff()  # 关闭画图窗口
            print(loss_cwt.item())
            print(loss_mar.item())
            # if loss1 < 35:  # 这里设置异常阈值，这里做一个参考
            #     print(loss1)
            #     print("出现异常，尽快处理")
            # else:
            #     # print(score.item())
            #     print("正常")
            # if loss1 < 42 or loss2 < 10:  # 这里设置异常阈值，这里做一个参考
            #     print(loss1.item())
            #     print(loss2.item())
            #     print("出现异常，尽快处理")
            # else:
            #     # print(score.item())
            #     print("正常")


def Dynamic_evaluate_model(model_path):
    rand_freeze = torch.tensor(np.load("seed.npy"))
    model = models.Vae_Model.load_from_checkpoint(model_path, in_channel=1, out_channel=8, tag="Conv_block2D", freeze_randn=rand_freeze)
    model.share_memory()
    model.eval()
    loss = nn.MSELoss().share_memory()
    pipe = Pipe()

    producer_process = Process(target=write_Sound_to_tensor, args=(pipe, ))  # 调用信号数据的生产者
    # producer_process = Process(target=write_Signal_to_tensor, args=(pipe, ))  # 调用音频数据的生产者
    consumer_process = Process(target=read_from_tensor, args=(pipe, model, loss))  # 调用消费者

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()


if __name__ == '__main__':
    model_path = "checkpoints/best-checkpoint-v12.ckpt"
    Dynamic_evaluate_model(model_path)
