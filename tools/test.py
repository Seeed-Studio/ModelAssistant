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

from tools.utils.config import load_config
from tools.utils.inference import Infernce
import edgelab.models
import edgelab.datasets
import edgelab.evaluation
import edgelab.engine
import edgelab.visualization


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('task',
                        choices=['cls', 'det', 'pose'],
                        help='task type')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--source',
        type=str,
        help='The data source to be tested, if not set, the verification data '
        'set test in the configuration file will be used by default')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument('--show',
                        action='store_true',
                        help='show prediction results')
    parser.add_argument('--show-dir',
                        help='directory where painted images will be saved. '
                        'If specified, it will be automatically saved '
                        'to the work_dir/timestamp/show_dir')
    parser.add_argument('--interval',
                        type=int,
                        default=1,
                        help='visualize per interval samples.')
    parser.add_argument('--wait-time',
                        type=float,
                        default=2,
                        help='the interval of show (s)')
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
    parser.add_argument('--tta', action='store_true')
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
    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        # cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        # cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator).append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    tmp_folder = tf.TemporaryDirectory()
    # Modify and create temporary configuration files
    config_data = load_config(args.config,
                              folder=tmp_folder.name,
                              cfg_options=args.cfg_options)
    # load temporary configuration files
    cfg = Config.fromfile(config_data)
    tmp_folder.cleanup()

    cfg = merge_args(cfg, args)  # pose
    cfg.launcher = args.launcher
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

    cfg.load_from = args.checkpoint

    if args.task == 'det' and (args.show or args.show_dir):
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta and args.task == 'det':

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(type='DetTTAModel',
                                 tta_cfg=dict(nms=dict(type='nms',
                                                       iou_threshold=0.5),
                                              max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(type='TestTimeAug',
                            transforms=[
                                [
                                    dict(type='RandomFlip', prob=1.),
                                    dict(type='RandomFlip', prob=0.)
                                ],
                                [
                                    dict(type='PackDetInputs',
                                         meta_keys=('img_id', 'img_path',
                                                    'ori_shape', 'img_shape',
                                                    'scale_factor', 'flip',
                                                    'flip_direction'))
                                ],
                            ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg or args.task == 'pose':
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.checkpoint.endswith(('.tflite', '.onnx', '.bin', '.param')):
        infer = Infernce(args.checkpoint,
                         dataloader=runner.test_dataloader,
                         cfg=cfg,
                         runner=runner,
                         source=args.source,
                         task=args.task,
                         show=args.show,
                         save_dir=args.show_dir)
        # Start Inference Testing
        infer.test()
    else:
        # add `DumpResults` dummy metric
        if args.task == 'det' and args.dump is not None:
            assert args.dump.endswith(('.pkl', '.pickle')), \
                'The dump file must be a pkl file.'
            runner.test_evaluator.metrics.append(
                DumpDetResults(out_file_path=args.dump))

        if args.task == 'pose' and args.dump:

            class SaveMetricHook(Hook):

                def after_test_epoch(self, _, metrics=None):
                    if metrics is not None:
                        mmengine.dump(metrics, args.dump)

            runner.register_hook(SaveMetricHook(), 'LOWEST')

        # start testing
        runner.test()


if __name__ == '__main__':
    main()