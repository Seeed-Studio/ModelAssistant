import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmdet.models.utils.misc import interpolate_as

import edgelab.core
import edgelab.models
import edgelab.datasets

from tools.utils.config import load_config

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('task',
                        default='mmdet',
                        choices=['mmcls', 'mmdet', 'mmpose'],
                        help='Choose training type')
    parser.add_argument('config',
                        default='configs/yolo/yolov3_mbv2_416_coco.py',
                        help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument('--auto-resume',
                        action='store_true',
                        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device',
                            help='device used for training. (Deprecated)')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-id',
                            type=int,
                            default=0,
                            help='id of gpu to use '
                            '(only applicable to non-distributed training)')
    parser.add_argument('--ipu-replicas',
                        type=int,
                        default=None,
                        help='num of ipu replicas to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
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
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',  #TODO
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument('--data', help='point data root manually')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def mkdir_work(work_dir):
    work_dir = osp.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    fl = os.listdir(work_dir)
    num = max([int(i.replace('exp', '')) for i in fl if 'exp' in i] + [0]) + 1
    os.makedirs(f'exp{num}', exist_ok=True)
    return osp.join(work_dir, f'exp{num}')


def main():
    # PWD = os.environ['PWD']
    # #check PWD in os.environ['PYTHONPATH']
    # if PWD not in os.environ['PYTHONPATH']:
    #     os.environ['PYTHONPATH'] += ':' + PWD

    args = parse_args()
    train_type = args.task
    config_data = load_config(args.config, args.cfg_options)
    cfg = Config.fromstring(config_data,
                            file_format=osp.splitext(args.config)[-1])
    if train_type == 'mmdet':
        from mmdet import __version__
        from mmdet.apis import init_random_seed, set_random_seed
        from mmdet.datasets import build_dataset
        from mmdet.models import build_detector as build_model
        from mmdet.utils import (collect_env, get_device, get_root_logger,
                                 setup_multi_processes, update_data_root)
        from tools.utils.config import replace_cfg_vals
        from edgelab.core.apis.mmdet.train import train_detector as train_model
        # replace the ${key} with the value of cfg.key
        cfg = replace_cfg_vals(cfg)
        # update data root according to MMDET_DATASETS
        update_data_root(cfg)
    elif train_type == 'mmcls':
        from mmcls import __version__
        from mmcls.apis import init_random_seed, set_random_seed, train_model
        from mmcls.datasets import build_dataset
        from mmcls.models import build_classifier as build_model
        from mmcls.utils import (auto_select_device, collect_env,
                                 get_root_logger, setup_multi_processes)
    else:
        from mmcv.runner import set_random_seed
        from mmpose import __version__
        from mmpose.apis import init_random_seed, train_model
        from mmpose.datasets import build_dataset
        from mmpose.models import build_posenet as build_model
        from mmpose.utils import collect_env, get_root_logger, setup_multi_processes

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.autoscale_lr:
        if 'autoscale_lr' in cfg and \
                'enable' in cfg.autoscale_lr and \
                'base_batch_size' in cfg.autoscale_lr:
            cfg.autoscale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

    # set multi-process settings
    setup_multi_processes(cfg)

    # get root dir
    root = osp.abspath('.')

    if args.data is not None:
        args.data = os.path.abspath(args.data)
        cfg.data_root = args.data
        cfg.data.train.data_root = args.data
        cfg.data.val.data_root = args.data
        cfg.data.test.data_root = args.data
    
    # convert to absolute paths
    if not os.path.isabs(cfg.data_root):
        cfg.data_root = os.path.abspath(cfg.data_root)
    if not os.path.isabs(cfg.data.train.data_root):
        cfg.data.train.data_root = os.path.abspath(cfg.data.train.data_root)
    if not os.path.isabs(cfg.data.val.data_root):
        cfg.data.val.data_root = os.path.abspath(cfg.data.val.data_root)
    if not os.path.isabs(cfg.data.test.data_root):
        cfg.data.test.data_root = os.path.abspath(cfg.data.test.data_root)      

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    else:
        # if cfg.work_dir is not none and the besename of work_dir is not equal to config filename, 
        # use config filename as the basename in cfg.work_dir. 
        if osp.splitext(osp.basename(args.config))[0] != osp.basename(cfg.work_dir):
            cfg.work_dir = osp.join(cfg.work_dir,
                                osp.splitext(osp.basename(args.config))[0])

  
    # turn relative path to absolute path for parameter load_from and resume_from
    if cfg.load_from and not osp.isabs(cfg.load_from):
        cfg.load_from = osp.join(root, cfg.load_from)

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if cfg.resume_from and not osp.isabs(cfg.resume_from):
        cfg.resume_from = osp.join(root, cfg.resume_from) 
    
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.work_dir = mkdir_work(cfg.work_dir)
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device() if train_type == 'mmdet' else (
        auto_select_device() if train_type == 'mmcls' else None)
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    if train_type == 'mmdet':
        model = build_model(cfg.model,
                            train_cfg=cfg.get('train_cfg'),
                            test_cfg=cfg.get('test_cfg'))
        model.init_weights()
    else:
        model = build_model(cfg.model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        assert 'val' in [mode for (mode, _) in cfg.workflow]
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.get(
            'pipeline', cfg.data.train.dataset.get('pipeline'))
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None and train_type == 'mmdet':
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(mmdet_version=__version__ +
                                          get_git_hash()[:7],
                                          CLASSES=datasets[0].CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = datasets[0].CLASSES
    # save mmcls version, config file content and class names in
    # runner as meta data
    if train_type == 'mmcls':
        meta.update(
            dict(mmcls_version=__version__,
                 config=cfg.pretty_text,
                 CLASSES=datasets[0].CLASSES))

    train_model(model,
                datasets,
                cfg,
                distributed=distributed,
                validate=(not args.no_validate),
                timestamp=timestamp,
                meta=meta)


if __name__ == '__main__':
    main()
