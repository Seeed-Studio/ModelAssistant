import argparse
import os
import os.path as osp
import warnings
import tempfile as tf
from copy import deepcopy

import mmengine
from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo

from tools.utils.inference import Inter, pfld_inference, audio_inference, show_point, fomo_inference
from tools.utils.config import load_config
import edgelab.models
import edgelab.datasets
import edgelab.evaluation
import edgelab.engine



def load_model(cfg, checkpoint, build_model, fuse=False):
    if checkpoint.endswith(('pt', 'pth')):
        model = build_model(cfg.model)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, checkpoint, map_location='cpu')
        if fuse:
            model = fuse_conv_bn(model)
        model.eval()
        pt = True
    else:
        if checkpoint.endswith(('.bin', '.param')):
            name = osp.basename(checkpoint)
            dir = osp.dirname(checkpoint)
            base = osp.splitext(name)[0]
            checkpoint = [
                osp.join(dir, i) for i in [base + '.bin', base + '.param']
            ]
        model = Inter(checkpoint)
        pt = False
    return model, pt

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('task',choices=['cls','det','pose'],help='task type')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args=parse_args()
    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()
    
    # pt model
    if args.config:
        