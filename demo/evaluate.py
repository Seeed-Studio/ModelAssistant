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
from model import Conv_Lstm
from matplotlib import pyplot as plt
from dataset_tool.tools import generate_gaf,min_max_scale, paa


def diff_block(data):
    zero_array = np.zeros((1, 3)).astype('float32')
    data_diff = np.diff(data.T, n=1).T
    data_diff = np.vstack((data_diff, zero_array))
    data = np.hstack((data, data_diff))
    return data

def psnr(loss):
    psnr = 20 * np.log10(1 / np.sqrt(loss))
    return psnr


def Signal_diff_process(data):
    data = np.reshape(np.array(data).astype('float32'), (-1, 3)) / 1500
    data = diff_block(data)
    return data


def Signal_process(data):
    data = np.reshape(np.array(data).astype('float32'), (-1, 3))
    temp = []
    for i in range(3):
        # gasf = generate_gaf(X, gaf_type='summation')
        data_temp = paa(data[:, i], 64)
        gadf_i = generate_gaf(data_temp, gaf_type='difference')
        # gadf_i = cv2.resize(gadf_i, (64,64), interpolation=cv2.INTER_LINEAR)
        temp.append(gadf_i)
    gadf = np.array(temp)
    return gadf



def Sound_process(data, mel_transform, db_transform):
    data = np.reshape(np.array(data).astype('float32'), (-1, ))
    data = torch.tensor(data)
    mel_spectrogram = mel_transform(data)
    mel_spectrogram_db = db_transform(mel_spectrogram)/100
    return mel_spectrogram_db.unsqueeze(0).numpy()


def write_Sound_to_tensor(produce, consume):
    mel_transform = MelSpectrogram(16000, n_fft=256, n_mels=32)
    db_transform = AmplitudeToDB()
    #produce, consume = pipe
    consume.close()
    ser = serial.Serial('COM5', 115200)
    if ser.isOpen():                        # 判断串口是否成功打开     
        print("打开串口成功。")     
        print(ser.name)    # 输出串口号
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


def write_Signal_to_tensor(produce, consume):
    
    #produce, consume = pipe
    consume.close()
    ser = serial.Serial('COM3', 115200)
    if ser.isOpen():                        # 判断串口是否成功打开     
        print("打开串口成功。")     
        print(ser.name)    # 输出串口号
    else:     
        print("打开串口失败。")
    while True:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        for index, i in enumerate(ser):
            if index == 0:
                continue
            data = i.decode().strip('\r\n').split(' ')[:-1]
            data = Signal_process(data)
            # if len(data) != 1024:
            #     continue
            produce.send(data)  # 将写入的信息放入管道

def read_from_tensor(pipe, model, loss):
    produce, consume = pipe
    produce.close()
    score_list = []
    while True:
        try:
            data = consume.recv()  # 从管道中获取写入的信息
            data_tensor = torch.tensor(data).unsqueeze(0)
            data_recon = model(data_tensor)
            score = loss(data_tensor, data_recon).item()
            score = psnr(score)
            score_list.append(score)
            # plt.clf()
            # plt.plot(score_list, 'r', label='Anomally Score')
            # plt.title('Anomally Score')
            # plt.legend()
            # # plt.show()
            # plt.pause(0.01)  # 暂停一段时间，不然画的太快会卡住显示不出来
            # plt.ioff()  # 关闭画图窗口
            print(score)
        #     if score.item()<=36:
        #         # print(score.item())
        #         print("出现异常，尽快处理")
        #     else:
        #         # print(score.item())
        #         print("正常")
        except EOFError:
            time.sleep(1)

# def evaluate_model():
#     data_root = "datasets"
#     loss = nn.MSELoss()
#     transform = transforms.Compose([transforms.ToTensor()])
#     test_dataset = Dataset_wrap(data_root=data_root, tag="Test", transform=transform)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#     model = Conv_Lstm.SimpleModel.load_from_checkpoint("checkpoints/best-checkpoint-v2.ckpt", in_channel=3, out_channel=64)
#     model.eval()
#     score_list = []
#     for item in test_loader:
#         item_recon = model(item)
#         score = loss(item_recon, item).item()
#         score_list.append(score)
#     # trainer = Trainer()
#     # loss = trainer.test(model=model, dataloaders=[test_loader])
#     return score_list

def Dynamic_evaluate_model(model_path):
    model = Conv_Lstm.SimpleModel.load_from_checkpoint(model_path, in_channel=3, out_channel=32, tag="Conv_block2D")
    model.share_memory()
    model.eval()
    loss = nn.MSELoss().share_memory()
    pipe = Pipe()

    # producer_process = Process(target=write_Signal_to_tensor, args=(pipe))
    producer_process = Process(target=write_Signal_to_tensor, args=(pipe))
    consumer_process = Process(target=read_from_tensor, args=(pipe, model, loss))
    

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()

if __name__ == '__main__':
    model_path = "checkpoints/best-checkpoint-v27.ckpt"
    Dynamic_evaluate_model(model_path)
