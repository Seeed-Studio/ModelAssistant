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
    produce, consume = pipe
    consume.close()
    ser = serial.Serial('COM8', 115200)
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
            if (index + 1) % dt.sample_rate != 0:
                temp = i.decode().strip('\r\n').split(' ')
                data = data + temp
                continue
            temp = i.decode().strip('\r\n').split(' ')
            data = data + temp
            # b = time.time()
            # print(b - a)
            # a = time.time()

            produce.send(data)  # 将写入的信息放入管道
            data = []
            ser.flushInput()
            # count = ser.inWaiting()
            # print(count)
            # b = time.time()
            # print(b - a)
            # a = time.time()


def read_from_tensor(pipe, model, loss):
    # mel_transform = MelSpectrogram(96000, n_fft=32, n_mels=16)
    # db_transform = AmplitudeToDB()
    produce, consume = pipe
    produce.close()
    data_mean = np.load('x_data_mean.npy')
    data_std = np.load('x_data_std.npy')
    # data_res = np.zeros((3, 64, 64))
    score_list = []
    while True:
        if consume.poll():
            data = consume.recv()  # 从管道中获取写入的信息
            data, c, label = dt.sample_c_process(data, data_mean, data_std)
            # data, c, label = dt.sample_mico_process(data, mel_transform, db_transform)
            label = torch.tensor(label).unsqueeze(0)
            # data_tensor_res = torch.tensor(data_res).unsqueeze(0)
            data_tensor = torch.tensor(data).unsqueeze(0)
            c = torch.tensor(c).unsqueeze(0)
            c_label = c
            input = torch.vstack((data_tensor, c))
            x, c = model(input)
            loss1, loss2 = tools.col_psnr(x, data_tensor, c, c_label)
            score_list.append(loss1.item())
            plt.clf()
            plt.plot(score_list, 'r', label='Anomally Score')
            plt.title('Anomally Score')
            plt.legend()
            # plt.show()
            plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
            plt.ioff()  # 关闭画图窗口
            # score1 = psnr(loss1.item())
            # score2 = psnr(loss2.item())
            # print(loss1.item())
            # print(loss2.item())
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
    model = models.Vae_Model.load_from_checkpoint(model_path, in_channel=3, out_channel=8, tag="Conv_block2D", freeze_randn=rand_freeze)
    model.share_memory()
    model.eval()
    loss = nn.MSELoss().share_memory()
    pipe = Pipe()

    # producer_process = Process(target=write_Sound_to_tensor, args=(pipe, ))
    producer_process = Process(target=write_Signal_to_tensor, args=(pipe, ))
    consumer_process = Process(target=read_from_tensor, args=(pipe, model, loss))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()


if __name__ == '__main__':
    model_path = "checkpoints/best-checkpoint-v2.ckpt"
    Dynamic_evaluate_model(model_path)
