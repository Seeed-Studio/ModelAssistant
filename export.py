import torch
import torchvision
import requests
import torch.onnx
import time
import torch
import torch.nn as nn
import serial
import numpy as np
from model import models
import sys
from tinynn.converter import TFLiteConverter
import tensorflow as tf
from tinynn.graph.quantization.quantizer import QATQuantizer
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def ckpt2tflite(data):

    # Provide a viable input to the model

    # Moving the model to cpu and set it to evaluation mode before model conversion
    with torch.no_grad():
        # model.cpu()
        # model.eval()

        converter = TFLiteConverter(model, data, optimize=5, tflite_path=tf_model_path)

        converter.convert()


x = torch.randn(2, 3, 32, 32).cpu()
model_path = "checkpoints/best-checkpoint-v137.ckpt"
rand_freeze = torch.tensor(np.load("seed.npy"))
model = models.Vae_Model.load_from_checkpoint(model_path, in_channel=3, out_channel=8, tag="Conv_block2D", freeze_randn=rand_freeze)
model.eval()
model.cpu()
model = model.to_torchscript(method="trace", example_inputs=x)
tf_model_path = 'test.tflite'
ckpt2tflite(x)
