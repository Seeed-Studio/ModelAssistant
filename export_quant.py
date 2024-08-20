import torch
import torchvision
import requests
import torch.onnx
import time
import torch
import torch.nn as nn
import serial
import numpy as np
from tqdm import tqdm
from model import models
import sys
from torch.utils.data import DataLoader

from tinynn.converter import TFLiteConverter

import os

import tensorflow as tf
from tinynn.graph.quantization.quantizer import QATQuantizer
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.graph.tracer import model_tracer
from dataset_tool.tools import Signal_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def calibrate(ptq_model):
    # TODO: Support multiple inputs
    # TODO: Support handle 'audio', 'sensor' inputs
    data_root = "datasets"
    ptq_model.eval()
    # for _ in range(10):
    #     dummy_input = torch.randn(2, 3, 32, 32)
    #     ptq_model(dummy_input)
    #     i = i + 1
    #     print(i)
    dataset = Signal_dataset(data_root=data_root, tag='Train')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    epoch = len(data_loader)
    dummy_input = torch.randn(2, 3, 32, 32)
    # data = torch.vstack((data[0], data[1]))
    # ptq_model(dummy_input)
    # print("估计完成")
    # epoch = 10
    with torch.no_grad(), tqdm(total=epoch, ncols=50) as pbar:
        for data in data_loader:
            dummy_input = torch.vstack((data[0], data[1]))
            # dummy_input = torch.randn(2, 3, 32, 32)
            # data = torch.vstack((data[0], data[1]))
            ptq_model(dummy_input)
            pbar.update(1)


def ckpt2tflite(data):
    # 参数设置
    algorithm = 'l2'
    work_dir = './'
    backend = 'qnnpack'

    # Provide a viable input to the model
    with model_tracer():
        # print(id(data))
        quantizer = PostQuantizer(
            model,
            data,
            work_dir=work_dir,
            config={
                'asymmetric': True,
                'set_quantizable_op_stats': True,
                'per_tensor': True,
                'algorithm': algorithm,
                'backend': backend,
            },
        )
        ptq_model = quantizer.quantize()
        ptq_model.cpu()

    calibrate(ptq_model)
    # ptq_model.to(device=context.device)
    # Moving the model to cpu and set it to evaluation mode before model conversion
    with torch.no_grad():
        ptq_model.eval()
        ptq_model = quantizer.convert(ptq_model)
        torch.backends.quantized.engine = quantizer.backend
        converter = TFLiteConverter(
            ptq_model,
            data,
            optimize=5,
            quantize_target_type='int8',
            fuse_quant_dequant=True,
            rewrite_quantizable=True,
            tflite_micro_rewrite=True,
            tflite_path=tf_model_path,
        )
        converter.convert()


def Sound_process(data, mel_transform, db_transform):
    data = np.reshape(np.array(data).astype('float32'), (-1, ))
    data = torch.tensor(data)
    mel_spectrogram = mel_transform(data)
    mel_spectrogram_db = db_transform(mel_spectrogram) / 100
    return mel_spectrogram_db.unsqueeze(0).unsqueeze(0).numpy()


x = torch.randn(2, 3, 32, 32, requires_grad=False).cpu()
model_path = "checkpoints/best-checkpoint-v4.ckpt"
rand_freeze = torch.tensor(np.load("seed.npy"))
model = models.Vae_Model.load_from_checkpoint(model_path, in_channel=3, out_channel=8, tag="Conv_block2D", freeze_randn=rand_freeze)
# model = model.to_torchscript(method="trace", example_inputs=x)

model.eval()
model.cpu()
tf_model_path = 'test_int8.tflite'
ckpt2tflite(x)
