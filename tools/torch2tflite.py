import argparse
import os
import tempfile as tf
import os.path as osp
import torch
import numpy as np
from copy import deepcopy
from functools import partial
import edgelab.models
import edgelab.datasets
import edgelab.evaluation
import edgelab.engine
from tools.utils.config import load_config

from tqdm import tqdm

from tensorflow import lite as tflite


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
    parser.add_argument('--algorithm', type=str, default='l2', choices=['l2', 'kl'])
    parser.add_argument('--backend', type=str, default='qnnpack', choices=['qnnpack', 'fbgemm'])
    parser.add_argument('--show', action='store_true',
                        help='show tflite graph')
    parser.add_argument(
        '--verify', action='store_true', default=False, help='verify the tflite model')
    parser.add_argument(
        '--simplify',
        type=int,
        default=5,
        help='level of graph simplification, 0 to 5')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        help='input data shape, e.g. 3 224 224')
    parser.add_argument('--epoch', type=int, default=100,
                        help='max epoch number of the quantization')
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

    return parser.parse_args()


def args_check():
    
    args = parse_args()

    
    """Check the args."""
    assert args.checkpoint is not None, 'Please specify the checkpoint file'
    assert args.type in ['int8', 'uint8', 'float32'], \
        'Please specify the tflite export type'
    assert args.config is not None, 'Please specify the config file'
    assert osp.exists(args.config), 'Config file does not exist'
    assert osp.exists(args.checkpoint), 'Checkpoint file does not exist'
    assert args.simplify >= 0 and args.simplify <= 5, 'Simplify level should be in [0, 5]'

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # load config
    tmp_folder = tf.TemporaryDirectory()
    # Modify and create temporary configuration files
    config_data = load_config(args.config,
                              folder=tmp_folder.name,
                              cfg_options=args.cfg_options)
    # load temporary configuration files
    cfg = Config.fromfile(config_data)
    tmp_folder.cleanup()
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    args.work_dir = cfg.work_dir
    
    cfg.val_dataloader['batch_size'] = 1
    cfg.val_dataloader['num_workers'] = 1
    
    if args.tflite_file is None:
        args.tflite_file = osp.join(args.work_dir, osp.basename(
            args.checkpoint) + '_' + args.type + '.tflite')

    try:
        if 'shape' in cfg:
            args.shape = cfg.shape
        elif 'width' in cfg and 'height' in cfg:
            args.shape = [
                3,
                cfg.width,
                cfg.height,
            ]
    except:
        raise ValueError('Please specify the input shape')

    return args, cfg


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
        epoch = min(len(context.val_loader), context.max_iteration)
        with tqdm(total=epoch, ncols=50) as pbar:
            for i, data in enumerate(context.val_loader):

                if context.max_iteration is not None and i >= context.max_iteration:
                    break
                #print('\rcalibration iteration: {}'.format(i+1), end='')
                inputs = data['inputs']
                if isinstance(inputs, (list, tuple)):
                    input = inputs[0] # only one input
                else:
                    input = inputs
                    
                assert isinstance(input, torch.Tensor), "The input should be a tensor"
                
                if input.dtype != torch.float32:
                    input = input / 255.0 # normalize to [0, 1]
                    
                if input.shape[0] != 1: 
                    input = input.unsqueeze(0) # batch size 1
                    
                input = input.to(device=context.device)
                model(input)

                pbar.update(1)
                context.iteration += 1

def export_tflite(args, model, context: DLContext):
    """Export the model to tflite
    Args:
        args: The args
        model: The model to be exported
        context (DLContext): The dataset context object
    """
    model.cpu().eval()
    
    #model.forward = partial(model.forward, mode='tensor')
    if len(args.shape) ==1 :
        dummy_input = torch.randn(1, args.shape)
    else:
        dummy_input = torch.randn(1, *args.shape)
    
    if args.type == 'int8' or type == 'uint8':
        with model_tracer():
            quantizer = PostQuantizer(model, dummy_input, work_dir=args.work_dir, config={
                                      'asymmetric': True, 'set_quantizable_op_stats': True, 'per_tensor': False, 'algorithm': args.algorithm, 'backend': args.backend})
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
                ptq_model, dummy_input, optimize=args.simplify,  quantize_target_type=args.type, fuse_quant_dequant=True, rewrite_quantizable=True,  tflite_micro_rewrite=True, tflite_path=args.tflite_file)

    else:
        with torch.no_grad():
            torch.backends.quantized.engine = 'qnnpack'
            converter = TFLiteConverter(
                model, dummy_input, optimize=args.simplify, tflite_path=args.tflite_file)

    converter.convert()
       
def main():
    
    args, cfg = args_check()
    
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    runner.call_hook('before_run')
    runner.load_checkpoint(args.checkpoint, map_location='cpu')

    context = DLContext()
    context.device = args.device
    context.val_loader = runner.val_dataloader
    context.max_iteration = args.epoch

    export_tflite(args, runner.model, context)


if __name__ == '__main__':
    main()
