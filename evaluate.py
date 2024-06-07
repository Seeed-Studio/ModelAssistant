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


def write_Sound_to_tensor(produce, consume):
    mel_transform = MelSpectrogram(16000, n_fft=256, n_mels=32)
    db_transform = AmplitudeToDB()
    consume.close()
    ser = serial.Serial('COM5', 115200)
    if ser.isOpen():  # 判断串口是否成功打开
        print("打开串口成功。")
        print(ser.name)  # 输出串口号
    else:
        print("打开串口失败。")
    while True:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        for index, i in enumerate(ser):
            if index == 0:
                continue
            data = i.decode().strip('\r\n').split(' ')[:-1]
            if len(data) != 4096:
                continue

            data = Sound_process(data, mel_transform, db_transform)

            produce.send(data)  # 将写入的信息放入管道


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
        for index, i in enumerate(ser):
            if (index + 1) % dt.sample_rate != 0:
                temp = i.decode().strip('\r\n').split(' ')
                data = data + temp
                continue
            temp = i.decode().strip('\r\n').split(' ')
            data = data + temp
            produce.send(data)  # 将写入的信息放入管道
            data = []


def read_from_tensor(pipe, model, loss):
    produce, consume = pipe
    produce.close()
    # data_res = np.zeros((3, 64, 64))
    # score_list = []
    while True:
        try:
            data = consume.recv()  # 从管道中获取写入的信息
            # print(data[0:3])
            # data, data_res, label, c = Signal_diff_process(data, data_res)
            data, c, label = dt.sample_c_process(data)
            label = torch.tensor(label).unsqueeze(0)
            # data_tensor_res = torch.tensor(data_res).unsqueeze(0)
            data_tensor = torch.tensor(data).unsqueeze(0)
            c = torch.tensor(c).unsqueeze(0)
            x_recon, c_recon, mu, mu_c, logvar, logvar_c, a, cluster_loss = model(data_tensor, c)
            loss1, loss2, loss3 = model.loss_function(x_recon, c_recon, data_tensor, c, mu, mu_c, logvar, logvar_c)
            # loss = loss1 / 20 + loss2 + loss3 + cluster_loss
            # loss1 = loss(data_recon[:, 0:3, :, :], label[:, 0:3, :, :]).item()
            # loss2 = loss(data_recon[:, 0:6, :, :], label[:, 0:6, :, :]).item()
            # data_recon, cluster, ker_loss = model(data_tensor, data_tensor_res, c)
            # score = loss(data_recon, label).item() + ker_loss.item()
            # a = a * 100
            # score = loss1 * a + loss2 * (1 - a)
            # plt.clf()
            # plt.plot(score_list, 'r', label='Anomally Score')
            # plt.title('Anomally Score')
            # plt.legend()
            # # plt.show()
            # plt.pause(0.01)  # 暂停一段时间，不然画的太快会卡住显示不出来
            # plt.ioff()  # 关闭画图窗口
            score1 = psnr(loss1.item())
            score2 = psnr(loss2.item())
            print(score1)
            print(score2)
            if score1 < 10 or score2 < 21.5:
                print(score1)
                print(score2)
                print("出现异常，尽快处理")
            else:
                # print(score.item())
                print("正常")
        except EOFError:
            time.sleep(1)


def Dynamic_evaluate_model(model_path):
    model = models.Vae_Model.load_from_checkpoint(model_path, in_channel=3, out_channel=8, tag="Conv_block2D")
    model.share_memory()
    model.eval()
    loss = nn.MSELoss().share_memory()
    pipe = Pipe()

    producer_process = Process(target=write_Signal_to_tensor, args=(pipe, ))
    consumer_process = Process(target=read_from_tensor, args=(pipe, model, loss))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()


if __name__ == '__main__':
    model_path = "checkpoints/best-checkpoint-v97.ckpt"
    Dynamic_evaluate_model(model_path)
