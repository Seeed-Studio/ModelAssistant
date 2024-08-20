import tensorflow as tf
import numpy as np
import torch
from model import models

test_arry = np.random.rand(2, 32, 32, 3).astype('float32')

# 加载ckpt模型并推理
model_path = "checkpoints/best-checkpoint-v110.ckpt"
rand_freeze = torch.tensor(np.load("seed.npy"))
model = models.Vae_Model.load_from_checkpoint(model_path, in_channel=3, out_channel=8, tag="Conv_block2D", freeze_randn=rand_freeze)
input = torch.tensor(test_arry)
input = input.permute(0, 3, 1, 2)
loss1, loss2 = model(input)

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path="test.tflite")

# 分配张量
interpreter.allocate_tensors()

# 获取输入和输出张量的信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 准备输入数据
# 根据模型的输入要求准备输入数据，这里假设输入是形状为 (1, 224, 224, 3) 的图像数据
input_data = test_arry

# 将输入数据设置到模型中
interpreter.set_tensor(input_details[0]['index'], input_data)

# 执行推理
interpreter.invoke()

# 获取输出数据
output_data = interpreter.get_tensor(output_details[1]['index'])
print(output_data)
print(loss2.item())
