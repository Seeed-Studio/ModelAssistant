import os
import cv2
import random
import argparse
import torchaudio
import torch.nn.functional as F
from models.tf.tf_common import *
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmpose.models import build_posenet
from mmcv import Config


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


def parse_args():
    parser = argparse.ArgumentParser(description='Convert PyTorch to TFLite')
    parser.add_argument('config', type=str, help='test config file path')
    parser.add_argument('--weights', type=str, help='torch model file path')
    parser.add_argument('--fp16', action='store_true', help='FP16 quantization for tflite')
    parser.add_argument('--data_root', type=str, help='Representative dataset path, need at least 100 images')
    parser.add_argument('--types', type=int, default=2, help='Type of representative dataset, '
                                                             '1: 1D data(audio), 2: 2D data(image)')
    parser.add_argument('--shape', type=int, nargs='+', default=[112], help='input data size')
    parser.add_argument('--classes', type=int, default=4, help='output numbers only for audio model')
    # parser.add_argument('--save', type=str, help='Tflite model path')

    args = parser.parse_args()

    return args


def representative_dataset_2d(root, size):
    assert os.path.exists(root), f'{root} not exists, please check out!'
    format = ['jpg', 'png', 'jpeg']
    for i, fn in enumerate(os.listdir(root)):
        if fn.split(".")[-1].lower() not in format:
            continue
        img = cv2.imread(os.path.join(root, fn))
        img = cv2.resize(img, (size[1], size[0]))[:, :, ::-1]
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img /= 255.0

        yield [img]
        if i >= 100:
            break


def representative_dataset_1d(root, size):
    assert os.path.exists(root), f'{root} not exists, please check out!'
    format = ['wav']
    for i, fn in enumerate(os.listdir(root)):
        if fn.split(".")[-1].lower() not in format:
            continue
        wave, sr = torchaudio.load(os.path.join(root, fn), normalize=True)
        wave = torchaudio.transforms.Resample(sr, 8000)(wave)
        wave.squeeze_()
        wave = (wave / wave.__abs__().max()).float()
        if wave.shape[0] >= size:
            max_audio_start = wave.size(0) - size
            audio_start = random.randint(0, max_audio_start)
            wave = wave[audio_start:audio_start + size]
        else:
            wave = F.pad(wave, (0, size - wave.size(0)),
                          "constant").data

        yield [wave[None, :, None]]
        if i > 100:
            break


def graph2dict(model):
    """Generating tf dictionary from torch graph."""

    gm: torch.fx.GraphModule = torch.fx.symbolic_trace(model, concrete_args={'flag': True})
    # gm.graph.print_tabular()
    modules = dict(model.named_modules())

    tf_dict = dict()
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            op = modules[node.target]
            if isinstance(op, nn.Conv2d):
                tf_dict[node.name] = TFBaseConv2d(op)
            if isinstance(op, nn.BatchNorm2d):
                tf_dict[node.name] = TFBN(op)
            if isinstance(op, nn.ReLU):
                tf_dict[node.name] = keras.layers.ReLU()
            if isinstance(op, nn.AdaptiveAvgPool2d):
                tf_dict[node.name] = keras.layers.GlobalAveragePooling2D()
            if isinstance(op, nn.Linear):
                tf_dict[node.name] = TFDense(op)
        if node.op == 'call_function':
            if 'add' in node.name:
                tf_dict[node.name] = keras.layers.Add()
            if 'cat' in node.name:
                tf_dict[node.name] = keras.layers.Concatenate(axis=node.args[1])

    return tf_dict, gm


class ModelParser(keras.layers.Layer):
    def __init__(self, tf_dict, gm):
        super().__init__()
        self.nodes = gm.graph.nodes
        self.tf = tf_dict

    def call(self, inputs):
        for node in self.nodes:
            if node.op == 'placeholder':
                globals()[node.name] = inputs
            if node.name in self.tf.keys():
                if 'add' in node.name:
                    globals()[node.name] = self.tf[node.name]([eval(str(x)) for x in node.args])
                elif 'cat' in node.name:
                    globals()[node.name] = self.tf[node.name]([eval(str(x)) for x in node.args[0]])
                else:
                    globals()[node.name] = self.tf[node.name](eval(str(node.args[0])))
            if 'size' in node.name:
                globals()[node.name] = eval(str(node.args[0])).shape[-1]
            if 'view' in node.name:
                s = eval(str(node.args[1]))
                globals()[node.name] = tf.keras.layers.Reshape([s])(eval(str(node.args[0])))
            if node.name == 'output':
                return eval(str(node.args[0]))


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


def tflite(keras_model, fp16, data_root, types):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if not fp16:
        if types == 2:
            input_shape = (keras_model.inputs[0].shape[1], keras_model.inputs[0].shape[2])
            converter.representative_dataset = lambda: representative_dataset_2d(data_root, input_shape)
        elif types == 1:
            input_shape = keras_model.inputs[0].shape[1]
            converter.representative_dataset = lambda: representative_dataset_1d(data_root, input_shape)
        else:
            raise Exception(f'{types} must be 1 or 2')
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.experimental_new_converter = True

    tflite_model = converter.convert()
    return tflite_model


def main(args):
    weights = os.path.abspath(args.weights)
    f = str(weights).replace('.pth', '_int8.tflite')
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_posenet(cfg.model)
    load_checkpoint(model, weights, map_location='cpu')
    model.cpu().eval()

    if args.types == 2:
        tf_dict, gm = graph2dict(model)
        mm = ModelParser(tf_dict, gm)
        imgsz = args.shape if len(args.shape) == 2 else [args.shape[0], args.shape[0]]
        inputs = tf.keras.Input(shape=(*imgsz, 3))
        outputs = mm(inputs)
        keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        keras_model.trainable = False
        # keras_model.summary()
    elif args.types == 1:
        keras_model = audio_keras(model, args.n_classes)
    tflite_model = tflite(keras_model, args.fp16, args.data_root, args.types)
    open(f, "wb").write(tflite_model)
    print(f'TFlite export sucess, saved as {f}')


if __name__ == '__main__':
    args = parse_args()
    main(args)