import requests
import torch
import torch.onnx
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import serial
import numpy as np
from torch.multiprocessing import Process, Pipe
from dataset_tool.tools import Signal_dataset
from model import models
from matplotlib import pyplot as plt
import onnx2tf
import onnx
import tensorflow as tf
from onnxsim import simplify

x1 = torch.randn(2, 3, 32, 32)
# x2 = torch.randn(1, 3, 32, 32)

model_tag = "Conv_block2D"
model_path = "checkpoints/best-checkpoint-v98.ckpt"
model = models.Vae_Model.load_from_checkpoint(model_path, in_channel=3, out_channel=8, tag=model_tag)
with torch.no_grad():
    torch.onnx.export(
        model,
        x1,
        "srcnn.onnx",
        opset_version=11,
        input_names=['input1'],
        output_names=['output1', 'output2'],
    )

model_simp, check = simplify(onnx.load("srcnn.onnx"))
onnx.checker.check_model(model_simp)
assert check, "assert check failed"
onnx.save(model_simp, "srcnn.onnx")
