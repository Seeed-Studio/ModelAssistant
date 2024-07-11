import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset_tool.tools import Signal_dataset, Microphone_dataset
from model import models
import misc
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Anomaly detection', add_help=False)
    parser.add_argument('--data_root', default="datasets", type=str, help="数据集的根路径")

    # Model parameters
    parser.add_argument('--in_channel', default=1, type=int, help="数据集的输入通道数")
    parser.add_argument('--out_channel', default=8, type=int, help="数据集的特征通道数")

    return parser


# Define a simple dataset (example using MNIST)
def Dataset_wrap(dataset_type, data_root, tag="Train", data_len=None, transform=None):
    if dataset_type == "Signal":
        dataset = Signal_dataset(data_root, tag, data_len=data_len)
    if dataset_type == "Micro":
        dataset = Microphone_dataset(data_root, tag, data_len=data_len)

    return dataset


def main():
    # 为了尽快出效果，这就都把参数写了之后记得打包这块的参数

    data_root = "datasets"
    in_channel = 3
    out_channel = 8
    dataset_tag = "Dynamic_Train"
    # dataset_tag = "Train"

    data_len = 100  # 指定动态读取数据时的总长度
    model_tag = "Conv_block2D"
    epochs = 500
    train_batch_size = 12
    val_batch_size = 4

    ####################################################
    # Data preparation
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_type = "Signal"
    dataset = Dataset_wrap(dataset_type, data_root=data_root, transform=transform, tag=dataset_tag, data_len=data_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
    # Model, callbacks, and trainer
    # model = Conv_Lstm.SimpleModel(in_channel=in_channel, out_channel=out_channel, tag=model_tag)

    rand_freeze = torch.tensor(np.load("seed.npy"))
    # rand_freeze = None
    model = models.Vae_Model(in_channel=in_channel, out_channel=out_channel, tag=model_tag, freeze_randn=rand_freeze)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 根据训练损失保存模型s
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min')
    # checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
    # early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')

    trainer = Trainer(max_epochs=epochs, callbacks=[checkpoint_callback])

    # Training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.save


if __name__ == '__main__':
    main()
