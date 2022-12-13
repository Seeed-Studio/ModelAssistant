import keras
import torch
import os
import cv2
import numpy as np
import random
import argparse
import torchaudio
import torch.nn.functional as F
import tensorflow as tf
from models.tf.tf_common import PFLDInference, Audio_Backbone,Audio_head


def parse_args():
    parser = argparse.ArgumentParser(description='Convert PyTorch to TFLite')
    parser.add_argument('--weights', type=str, help='torch model file path')
    parser.add_argument('--data_root', type=str, help='Representative dataset path, need at least 100 images')
    parser.add_argument('--name', type=str, help='model name that needs to be converted to tflite, '
                                                 'supported: pfld, audio, yolo')
    parser.add_argument('--shape', type=int, nargs='+', default=[112], help='input data size')
    parser.add_argument('--classes', type=int, default=4, help='output numbers only for audio model')
    # parser.add_argument('--save', type=str, help='Tflite model path')

    args = parser.parse_args()

    return args


def representative_dataset_pfld(img_root, img_size):
    assert os.path.exists(img_root), f'{img_root} not exists, please check out!'
    format = ['jpg', 'png', 'jpeg']
    for i, fn in enumerate(os.listdir(img_root)):
        if fn.split(".")[-1].lower() not in format:
            continue
        img = cv2.imread(os.path.join(img_root, fn))
        img = cv2.resize(img, (img_size[1], img_size[0]))[:, :, ::-1]
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img /= 255.0

        yield [img]
        if i >= 100:
            break


def representative_dataset_audio(root, size):
    assert os.path.exists(root), f'{root} not exists, please check out!'
    format = ['wav']
    for i, fn in enumerate(os.listdir(root)):
        if fn.split(".")[-1].lower() not in format:
            continue
        wave, sr = torchaudio.load(os.path.join(root, fn), normalize=True)
        wave = torchaudio.transforms.Resample(sr, 8000)(wave)
        wave.squeeze_()
        wave = (wave / wave.__abs__().max()).float()
        if wave.shape[0] >= 8192:
            max_audio_start = wave.size(0) - size
            audio_start = random.randint(0, max_audio_start)
            wave = wave[audio_start:audio_start + size]
        else:
            wave = F.pad(wave, (0, size - wave.size(0)),
                          "constant").data

        yield [wave[None, :, None]]
        if i > 100:
            break


def pfld_keras(model, shape):
    names = [['backbone.conv1', 'backbone.bn1'],
             ['backbone.conv2', 'backbone.bn2'],
             'backbone.conv3_1',
             'backbone.block3_2',
             'backbone.block3_3',
             'backbone.block3_4',
             'backbone.block3_5',
             'backbone.conv4_1',
             'backbone.conv5_1',
             'backbone.block5_2',
             'backbone.block5_3',
             'backbone.block5_4',
             'backbone.block5_5',
             'backbone.block5_6',
             'backbone.conv6_1',
             ['backbone.conv7.0', 'backbone.conv7.1'],
             'backbone.conv8',
             'backbone.fc',]
    tfout = PFLDInference(names=names, w=model)

    imgsz = shape if len(shape) == 2 else [shape[0], shape[0]]
    inputs = tf.keras.Input(shape=(*imgsz, 3))
    outputs = tfout(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    # keras_model.summary()

    return keras_model


def audio_keras(model, n_classes=4, size=8192):
    backbone = Audio_Backbone(nf=2, clip_length=64, factors=[4, 4, 4], out_channel=36, w=model)
    head = Audio_head(in_channels=36, n_classes=n_classes, w=model)
    tfout = keras.Sequential([backbone, head])

    inputs = tf.keras.Input(shape=(size, 1))
    outputs = tfout(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    # keras_model.summary()

    return keras_model


def tflite(keras_model, int8, data_root, name):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8:
        if name == 'pfld':
            input_shape = (keras_model.inputs[0].shape[1], keras_model.inputs[0].shape[2])
            converter.representative_dataset = lambda: representative_dataset_pfld(data_root, input_shape)
        elif name == 'audio':
            input_shape = keras_model.inputs[0].shape[1]
            converter.representative_dataset = lambda: representative_dataset_audio(data_root, input_shape)
        else:
            raise Exception(f'{name} must in ["pfld", "audio"]')
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.experimental_new_converter = True

    tflite_model = converter.convert()
    return tflite_model


def main(args):
    weights = args.weights
    data_root = args.data_root
    name = args.name
    shape = args.shape
    n_classes = args.classes

    weights = os.path.abspath(weights)
    f = str(weights).replace('.pth', '_int8.tflite')
    torch_model = torch.load(weights, map_location=torch.device('cpu'))
    model = torch_model['state_dict']
    if name == 'pfld':
        keras_out = pfld_keras(model, shape)
    elif name == 'audio':
        keras_out = audio_keras(model, n_classes=n_classes)
    tflite_model = tflite(keras_out, int8=True, data_root=data_root, name=name)
    open(f, "wb").write(tflite_model)
    print(f'TFlite export sucess, saved as {f}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
