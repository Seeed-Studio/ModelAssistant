import os
import cv2
import random
import argparse
import torchaudio
import torch.nn.functional as F
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmcv import Config, DictAction
import os.path as osp

import edgelab.models
import edgelab.datasets
import edgelab.core
from edgelab.models.tf.tf_common import *
from edgelab.core.utils.helper_funcs import representative_dataset, check_type
from tools.utils.config import load_config

try:
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    tf.config.experimental.set_visible_devices([], 'GPU')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert PyTorch to TFLite')
    parser.add_argument('task',
                        default='mmdet',
                        choices=['mmcls', 'mmdet', 'mmpose'],
                        help='Choose training task type')
    parser.add_argument('config',
                        type=str,
                        default='',
                        help='test config file path')
    parser.add_argument('--weights', type=str, help='torch model file path')
    parser.add_argument('--tflite_type',
                        type=str,
                        default='fp32',
                        help='Quantization type for tflite, '
                        '(int8, fp16, fp32)')
    # parser.add_argument(
    #     '--data',
    #     type=str,
    #     help='Representative dataset path, need at least 100 images')
    parser.add_argument('--audio',
                        action='store_true',
                        help='Choose audio dataset load code if given')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    # parser.add_argument('--shape',
    #                     type=int,
    #                     nargs='+',
    #                     default=[112],
    #                     help='input data size')
    # parser.add_argument('--save', type=str, help='Tflite model path')

    args = parser.parse_args()

    return args


def graph2dict(model):
    """Generating tf dictionary from torch graph."""
    
    #print("model: ", model)

    gm: torch.fx.GraphModule = torch.fx.symbolic_trace(
        model, concrete_args={'flag': True})
    #gm.graph.print_tabular()
    modules = dict(model.named_modules())

    tf_dict = dict()
    for node in gm.graph.nodes:
        if node.op == 'get_attr':
            tf_dict[node.name] = modules[node.target.rsplit('.', 1)[0]] # Audio model
        elif node.op == 'call_module':
            op = modules[node.target]
            if isinstance(op, (nn.Conv2d, nn.Conv1d)):
                tf_dict[node.name] = eval(f'TFBase{op.__class__.__name__}')(op)
            elif isinstance(op, (nn.BatchNorm2d, nn.BatchNorm1d)):
                tf_dict[node.name] = TFBN(op)
            elif isinstance(op, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.Sigmoid)):
                tf_dict[node.name] = TFActivation(op)
            elif 'drop_path' in node.name:  # yolov3
                tf_dict[node.name] = tf.identity
            elif isinstance(op, nn.Dropout):
                tf_dict[node.name] = keras.layers.Dropout(rate=op.p)

            elif isinstance(op, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d)):
                tf_dict[node.name] = tf_pool(op)
            elif isinstance(op, nn.Linear):
                tf_dict[node.name] = TFDense(op)
            elif isinstance(op, nn.Softmax) or node.name == 'sm':
                tf_dict[node.name] = keras.layers.Softmax(axis=-1)
            elif isinstance(op, nn.Upsample):
                # tf_dict[node.name] = keras.layers.UpSampling2D(size=int(op.scale_factor), interpolation=op.mode)
                tf_dict[node.name] = TFUpsample(op)
            else:
                pass
        elif node.op == 'call_function':
            if 'add' in node.name:
                tf_dict[node.name] = keras.layers.Add()
                # tf_dict[node.name] = lambda y: keras.layers.add([eval(str(x)) for x in y.args])
            elif 'cat' in node.name:
                dim = node.args[1] if len(
                    node.args) == 2 else node.kwargs['dim']
                dim = -1 if dim == 1 else dim - 1
                tf_dict[node.name] = keras.layers.Concatenate(axis=dim)
                # tf_dict[node.name] = lambda y: tf.concat([eval(str(x)) for x in y.args[0]], axis=dim)
            elif 'conv1d' in node.name:  # audio model
                tf_dict[node.name] = TFAADownsample(tf_dict[str(node.args[1])])
            elif 'interpolate' in node.name:
                tf_dict[node.name] = TFUpsample(node)
            elif 'view' in node.name:
                tf_dict[node.name] = TFView(node)
            elif 'relu' in node.name:
                tf_dict[node.name] = keras.layers.ReLU()
            elif 'softmax' in node.name:
                tf_dict[node.name] = keras.layers.Softmax(axis=-1)
            else:
                pass
        else:
            pass

    return tf_dict, gm


class ModelParser(keras.layers.Layer):

    def __init__(self, tf_dict, gm):
        super().__init__()
        self.nodes = gm.graph.nodes
        self.tf = tf_dict

    def call(self, inputs):
        for node in self.nodes:
            if node.op == 'get_attr':  # Fixed weight, pass
                continue
            elif node.op == 'placeholder':
                globals()[node.name] = inputs
            else:
                if node.name in self.tf.keys():
                    if 'add' in node.name:
                        globals()[node.name] = self.tf[node.name](
                            [eval(str(x)) for x in node.args])
                    elif 'cat' in node.name:
                        globals()[node.name] = self.tf[node.name](
                            [eval(str(x)) for x in node.args[0]])
                    else:
                        globals()[node.name] = self.tf[node.name](eval(
                            str(node.args[0])))
                elif 'size' in node.name:
                    if len(node.args) == 2:
                        # dim = -1 if node.args[1] == 1 else node.args[1]
                        globals()[node.name] = eval(str(node.args[0])).shape[-1]
                    else:
                        n, h, w, c = eval(str(node.args[0])).shape.as_list()
                        globals()[node.name] = [n, c, h, w]
                elif 'getitem' in node.name:
                    globals()[node.name] = eval(str(node.args[0]))[node.args[1]]
                elif 'floordiv' in node.name:
                    globals()[node.name] = eval(str(node.args[0])) // 2
                elif 'view' in node.name:
                    if len(node.args[1:]) == 2:
                        s = eval(str(node.args[1]))
                        globals()[node.name] = tf.reshape(eval(str(node.args[0])),
                                                        [-1, s])
                    else:
                        if 'contiguous' in str(node.args[0]):
                            n, c, h, w = [eval(str(a)) for a in node.args[1:]]
                            globals()[node.name] = tf.reshape(
                                (eval(str(node.args[0]))), [-1, h, w, c])
                        else:
                            n, group, f, h, w = [
                                a if isinstance(a, int) else eval(str(a))
                                for a in node.args[1:]
                            ]
                            globals()[node.name] = tf.reshape(
                                (eval(str(node.args[0]))), [-1, h, w, group, f])
                elif 'transpose' in node.name:
                    globals()[node.name] = tf.transpose(eval(str(node.args[0])),
                                                        perm=[0, 1, 2, 4, 3])
                elif 'contiguous' in node.name:
                    globals()[node.name] = eval(str(node.args[0]))
                elif 'mul' in node.name:
                    globals()[node.name] = node.args[0] * eval(str(node.args[1]))
                elif 'chunk' in node.name:
                    globals()[node.name] = tf.split(eval(str(node.args[0])),
                                                    node.args[1],
                                                    axis=-1)
                elif 'flatten' in node.name:
                    globals()[node.name] = tf.keras.layers.Flatten()(eval(str(node.args[0])))
                elif 'output' == node.name:
                    return eval(str(node.args[0]))
                else:
                    pass


def tflite(keras_model, type, data):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if type == 'fp16':
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    if type == 'int8':
        converter.representative_dataset = lambda: representative_dataset(data)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.experimental_new_converter = True

    tflite_model = converter.convert()
    return tflite_model


def main():
    args = parse_args()
    weights = os.path.abspath(args.weights)
    f = str(weights).replace('.pth', f'_{args.tflite_type}.tflite')

    config_data = load_config(args.config, args.cfg_options)
    cfg = Config.fromstring(config_data,
                            file_format=osp.splitext(args.config)[-1])
    cfg.model.pretrained = None

    # Get shape and datataloader
    _, build_model, build_dataset, build_dataloader = check_type(args.task)
    
    try:
        if cfg.shape is not None:
            shape = cfg.shape
        else:
            shape = [cfg.height, cfg.width, 3] if not args.audio else [cfg.width, 1]
    except:
        try:
            shape = [cfg.height, cfg.width, 3] if not args.audio else [cfg.width, 1]
        except:
            raise ValueError('Please specify the shape of the input data in the config file')
    
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))

    # build model
    model = build_model(cfg.model)
    load_checkpoint(model, weights, map_location='cpu')
    model.cpu().eval()

    tf_dict, gm = graph2dict(model)
    mm = ModelParser(tf_dict, gm)
    inputs = tf.keras.Input(shape=shape)
    outputs = mm(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    tflite_model = tflite(keras_model, args.tflite_type, dataset)
    open(f, "wb").write(tflite_model)
    print(f'TFlite export sucess, saved as {f}')


if __name__ == '__main__':
    main()