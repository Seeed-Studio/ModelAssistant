import argparse
import os
import os.path as osp
from copy import deepcopy
import edgelab.models
import edgelab.datasets
import edgelab.evaluation
import edgelab.engine
import edgelab.visualization

import torch
from mmengine.analysis import get_model_complexity_info
from mmengine.config import Config, DictAction, ConfigDict
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from mmdet.utils import setup_cache_size_limit_of_dynamo

from tools.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('task',
                        default='mmdet',
                        choices=['cls', 'det', 'pose'],
                        help='Choose training type')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--amp',
                        action='store_true',
                        default=False,
                        help='enable automatic-mixed-precision training')
    parser.add_argument('--auto-scale-lr',
                        action='store_true',
                        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    # vision
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument('--interval',
                        type=int,
                        default=1,
                        help='visualize per interval samples.')
    parser.add_argument('--wait-time',
                        type=float,
                        default=1,
                        help='display time of every window. (second)')

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # visualization-
    if args.task == 'pose' and (args.show or (args.show_dir is not None)):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.task == 'cls':
        # set dataloader args
        default_dataloader_cfg = ConfigDict(
            pin_memory=True,
            persistent_workers=True,
            collate_fn=dict(type='default_collate'),
        )
        if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
            default_dataloader_cfg.persistent_workers = False

        def set_default_dataloader_cfg(cfg, field):
            if cfg.get(field, None) is None:
                return
            dataloader_cfg = deepcopy(default_dataloader_cfg)
            dataloader_cfg.update(cfg[field])
            cfg[field] = dataloader_cfg
            if args.no_pin_memory:
                cfg[field]['pin_memory'] = False
            if args.no_persistent_workers:
                cfg[field]['persistent_workers'] = False

        set_default_dataloader_cfg(cfg, 'train_dataloader')
        set_default_dataloader_cfg(cfg, 'val_dataloader')
        set_default_dataloader_cfg(cfg, 'test_dataloader')

    return cfg


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    import tempfile as tf

    # load config
    tmp_folder = tf.TemporaryDirectory()
    # Modify and create temporary configuration files
    config_data = load_config(args.config,
                              folder=tmp_folder.name,
                              cfg_options=args.cfg_options)
    # load temporary configuration files
    cfg = Config.fromfile(config_data)
    tmp_folder.cleanup()
    cfg = merge_args(cfg, args)

    # set preprocess configs to model
    if 'preprocess_cfg' in cfg:
        cfg.model.setdefault('data_preprocessor',
                             cfg.get('preprocess_cfg', {}))

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # model complex anly
    try:
        if 'shape' in cfg:
            shape = cfg.shape
        elif 'width' in cfg and 'height' in cfg:
            shape = [
                3,
                cfg.width,
                cfg.height,
            ]
    except:
        raise ValueError('Please specify the input shape')

    if type(shape) == int:
        inputs = torch.rand(1, shape)
    else:
        inputs = torch.rand(1, *shape)
        
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        runner.model.cuda()
        runner.model.eval()
    analysis_results = get_model_complexity_info(model=runner.model,
                                                 input_shape=shape,
                                                 inputs=(inputs, ))
    print('=' * 30)
    print(f"Model Flops:{analysis_results['flops_str']}")
    print(f"Model Parameters:{analysis_results['params_str']}")
    print('=' * 30)

    # start training
    runner.train()


if __name__ == '__main__':
    main()