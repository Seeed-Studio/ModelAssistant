import argparse
import logging
import os
import os.path as osp
import torch
import numpy as np
from copy import deepcopy
import edgelab.models
import edgelab.datasets
import edgelab.evaluation
import edgelab.engine

from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config, DictAction, ConfigDict
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

from mmdet.utils import setup_cache_size_limit_of_dynamo

from tinynn.converter import TFLiteConverter
from tinynn.util.train_util import DLContext
from tinynn.graph.tracer import model_tracer
from tinynn.graph.quantization.quantizer import PostQuantizer

import tflite_runtime.interpreter as tflite


def parse_args():
    """Parse args from command line."""
    parser = argparse.ArgumentParser(description='Convert Pytorch to TFLite')
    parser.add_argument('config', help='test config file path')
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument('--tflite-file', help='tflite file', default=None)
    parser.add_argument('--checkpoint', help='checkpoint file', default=None)
    parser.add_argument('--device', type=str, help='device used for inference')
    parser.add_argument('--type', type=str, default='int8',
                        help='tflite export type', choices=['int8', 'uint8', 'float32'])
    parser.add_argument('--show', action='store_true',
                        help='show tflite graph')
    parser.add_argument(
        '--verify', action='store_true', default=False, help='verify the tflite model')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        help='input data shape, e.g. 3 224 224')
    parser.add_argument('--epoch', type=int, default=100,
                        help='max epoch number of the quantization')
    args = args_check(parser.parse_args())
    return args


def args_check(args):
    """Check the args."""
    assert args.checkpoint is not None, 'Please specify the checkpoint file'
    assert args.type in ['int8', 'uint8', 'float32'], \
        'Please specify the tflite export type'
    assert args.config is not None, 'Please specify the config file'
    assert osp.exists(args.config), 'Config file does not exist'
    assert osp.exists(args.checkpoint), 'Checkpoint file does not exist'
    assert args.work_dir is None or osp.exists(args.work_dir), \
        'Please specify the work dir'

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = Config.fromfile(args.config)

    if args.work_dir is None:
        args.work_dir = osp.join(osp.abspath(
            osp.dirname(args.checkpoint)), 'tflite')

    if args.tflite_file is None:
        args.tflite_file = osp.join(args.work_dir, osp.basename(
            args.checkpoint) + '_' + args.type + '.tflite')

    try:
        args.shape = [
            3,
            cfg.height,
            cfg.width,
        ]
    except:
        raise ValueError('Please specify the input shape')

    return args


def calibrate(model, context: DLContext):
    """Calibrates the fake-quantized model
    Args:
        model: The model to be validated
        context (DLContext): The context object
    """

    model.to(device=context.device)
    model.eval()
    context.iteration = 0
    with torch.no_grad():
        for i, data in enumerate(context.val_loader):

            if context.max_iteration is not None and i >= context.max_iteration:
                break
            print('\rcalibration iteration: {}'.format(i), end='')
            inputs = data['inputs']
            inputs = inputs.to(device=context.device)
            model(inputs)

            context.iteration += 1


def verify_tflite(args, context: DLContext):
    """Verify the tflite model
    Args:
        args: The args
        context (DLContext): The dataset context object
    """
    tflite_model = tflite.Interpreter(
        model_path=args.tflite_file)
    tflite_model.allocate_tensors()
    tflite_input = tflite_model.get_input_details()
    tflite_output = tflite_model.get_output_details()

    for i, data in enumerate(context.val_loader):
        if context.max_iteration is not None and i >= context.max_iteration:
            break
        image = data['inputs']
        input = image.unsqueeze(0).numpy().astype(
            np.int8).transpose(0, 2, 3, 1)
        tflite_model.set_tensor(tflite_input[0]['index'], input)
        tflite_model.invoke()
        output = tflite_model.get_tensor(tflite_output[0]['index'])
        print(output)


def export_tflite(args, model, context: DLContext):
    """Export the model to tflite
    Args:
        args: The args
        model: The model to be exported
        context (DLContext): The dataset context object
    """
    dummy_input = torch.randn(1, *args.shape)
    if args.type == 'int8' or type == 'uint8':
        with model_tracer():
            quantizer = PostQuantizer(model, dummy_input, work_dir=args.work_dir, config={
                                      'asymmetric': True, 'set_quantizable_op_stats': True, 'per_tensor': False})
            ptq_model = quantizer.quantize()
            ptq_model.to(device=context.device)

        calibrate(ptq_model, context)

        with torch.no_grad():
            ptq_model.cpu()
            ptq_model.eval()

            # The step below converts the model to an actual quantized model, which uses the quantized kernels.
            ptq_model = quantizer.convert(ptq_model)

            # When converting quantized models, please ensure the quantization backend is set.
            torch.backends.quantized.engine = quantizer.backend

            converter = TFLiteConverter(
                ptq_model, dummy_input, quantize_target_type=args.type, fuse_quant_dequant=True, rewrite_quantizable=True, tflite_path=args.tflite_file)

    else:
        with torch.no_grad():
            model.cpu()
            model.eval()
            torch.backends.quantized.engine = 'qnnpack'
            converter = TFLiteConverter(
                model, dummy_input, tflite_path=args.tflite_file)

    converter.convert()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    runner.load_checkpoint(args.checkpoint, map_location='cpu')

    context = DLContext()
    context.device = args.device
    context.val_loader = runner.val_dataloader
    context.max_iteration = args.epoch

    export_tflite(args, runner.model, context)

    if args.verify:
        verify_tflite(args, context)


if __name__ == '__main__':
    main()
