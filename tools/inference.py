# copyright Copyright (c) Seeed Technology Co.,Ltd.
import argparse
import os
import os.path as osp
import sys
import tempfile

import torch

current_path = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(current_path))

# TODO: Move to config file
import sscma.datasets  # noqa
import sscma.engine  # noqa
import sscma.evaluation  # noqa
import sscma.models  # noqa
import sscma.visualization  # noqa


def parse_args():
    from mmengine.config import DictAction

    parser = argparse.ArgumentParser(description='Test and Inference a trained model')

    # common configs
    parser.add_argument('config', type=str, help='the model config file path')
    parser.add_argument('checkpoint', type=str, help='the checkpoint file path')
    parser.add_argument(
        '--task',
        type=str,
        default='auto',
        choices=['auto', 'mmcls', 'mmdet', 'mmpose'],
        help='the task type of the model',
    )
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='The data source to be tested, if not set, the verification dataset in the configuration file will be used by default',
    )
    parser.add_argument(
        '--work_dir',
        '--work-dir',
        type=str,
        default=None,
        help='the directory to save logs and models',
    )
    parser.add_argument(
        '--dump',
        type=str,
        default=None,
        help='the path for a pickle dump of predictions',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='the device used for inference',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='show prediction results window',
    )
    parser.add_argument(
        '--out_dir',
        '--out-dir',
        type=str,
        default=None,
        help='the folder path to save prediction results ',
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='the interval of visualization per samples',
    )
    parser.add_argument(
        '--wait_time',
        '--wait-time',
        type=float,
        default=0.03,
        help='the visualize duration (seconds) of each sample',
    )
    parser.add_argument(
        '--input_type',
        '--input-type',
        type=str,
        default='image',
        choices=['audio', 'image', 'text'],
        help='the input type of model',
    )
    parser.add_argument(
        '--launcher',
        type=str,
        default='none',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        help='the job launcher for MMEngine',
    )
    parser.add_argument(
        '--cfg_options',
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help="override some settings in the used config, the key-value pair in 'xxx=yyy' format will be merged into config file",
    )

    # Detection specific
    parser.add_argument(
        '--tta',
        action='store_true',
        default=False,
        help='Detection: use test time augmentation (https://mmdetection.readthedocs.io/en/latest/user_guides/test.html#test-time-augmentation-tta)',
    )

    return parser.parse_args()


def verify_args(args):
    assert os.path.splitext(args.config)[-1] == '.py', "The config file name should be ended with a '.py' extension"
    assert os.path.exists(args.config), 'The config file does not exist'
    assert os.path.splitext(args.checkpoint)[-1] in {
        '.pth',
        '.onnx',
        '.tflite',
        '.param',
        '.bin',
    }, "The checkpoint model should be ended with a '.pth', '.onnx' or '.tflite' extension"
    assert os.path.exists(args.checkpoint), 'The checkpoint model does not exist'
    assert args.interval > 0, 'The interval of visualization per samples should be larger than 0'
    assert args.wait_time >= 0, 'The visualize duration should be larger than or equal to 0'

    return args


def get_exp_from_config(file_path, var_name):
    # NOTE: val = [exp] -> val = {exp}
    brk = {'}': '{', ']': '['}
    mrk = {'"': '"', "'": "'"}
    sta = False
    stk = list()
    res = set()
    with open(file_path, mode='r', encoding='utf-8') as file:
        while True:
            line = file.readline()
            if len(line) == 0:
                break
            line = line.strip()
            if len(stk) == 0:
                if not line.startswith(var_name):
                    continue
                line = line.replace(var_name, '', 1)
                res = set()
            idx = 0
            exp = str()
            while idx < len(line):
                ch = line[idx]
                if not sta:
                    if ch == '#':
                        break
                    elif ch in brk.values() or ch in mrk.values():
                        sta = True if ch in mrk.values() else sta
                        stk.append(ch)
                    elif ch in brk.keys():
                        if stk.pop() != brk[ch]:
                            raise RuntimeError
                else:
                    if ch in mrk.keys():
                        if stk[-1] != mrk[ch]:
                            exp += ch
                        else:
                            stk.pop()
                            sta = False
                            res.add(exp)
                            exp = str()
                    elif ch != '\\':
                        exp += ch
                    else:
                        idx += 1
                        exp += line[idx]
                idx += 1
            if len(stk) == 0:
                return [v for v in res if len(v) != 0]
    return []


def get_task_from_config(config_path):
    # TODO: currently the syntax like b = ['config'], a = b is not supported
    # TODO: support multiple occurrence
    base_type = {
        'default_runtime_cls.py': 'mmcls',
        'default_runtime_det.py': 'mmdet',
        'default_runtime_pose.py': 'mmpose',
    }
    config_name = os.path.basename(config_path)
    if config_name in base_type.keys():
        return [base_type[config_name]]
    exp = get_exp_from_config(config_path, '_base_')
    if len(exp) != 0:
        res = []
        for p in exp:
            res.extend(get_task_from_config(os.path.join(os.path.dirname(config_path), p)))
        return res
    return []


def build_config(args):
    from mmengine.config import Config

    from sscma.utils import load_config

    if args.task == 'auto':
        task = {'mmcls', 'mmdet', 'mmpose'}.intersection(get_task_from_config(args.config))
        assert len(task) == 1, 'Unable to get task from configs, please manually specify in arguments'
        args.task = list(task)[0]
        print('Using task type from config: {}'.format(args.task))

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_data = load_config(args.config, folder=tmp_dir, cfg_options=args.cfg_options)
        cfg = Config.fromfile(cfg_data)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.val_dataloader['batch_size'] = 1
    cfg.val_dataloader['num_workers'] = 1

    if 'batch_shapes_cfg' in cfg.val_dataloader.dataset:
        cfg.val_dataloader.dataset.batch_shapes_cfg = None

    cfg.launcher = args.launcher

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        args.work_dir = cfg.work_dir = os.path.join('work_dirs', os.path.splitext(os.path.basename(args.config))[0])

    if args.show or (args.out_dir is not None):
        assert 'visualization' in cfg.default_hooks, "VisualizationHook is required in 'default_hooks'"
        if args.task == 'mmcls':
            cfg.default_hooks.visualization.enable = True
        else:
            cfg.default_hooks.visualization.draw = True
        cfg.default_hooks.visualization.interval = args.interval
    if args.show:
        cfg.default_hooks.visualization.show = True
        cfg.default_hooks.visualization.wait_time = args.wait_time
    if args.out_dir is not None:
        cfg.default_hooks.visualization.out_dir = args.out_dir

    if args.dump is None:
        args.dump = args.checkpoint.replace(os.path.splitext(args.checkpoint)[-1], '.pkl')
        print('Using dump path from checkpoint: {}'.format(args.dump))

    if args.dump is not None:
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator).append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    if args.device.startswith('cuda'):
        args.device = args.device if torch.cuda.is_available() else 'cpu'

    if args.tta:
        if 'tta_model' not in cfg:
            raise RuntimeError("Cannot find 'tta_model' in config")

        if 'tta_pipeline' not in cfg:
            raise RuntimeError("Cannot find 'tta_pipeline' in config")

    return args, cfg


def main():
    args = parse_args()
    args = verify_args(args)
    args, cfg = build_config(args)

    if 'runner_type' not in cfg:
        from mmengine.runner import Runner

        runner = Runner.from_cfg(cfg)
    else:
        from mmengine.registry import RUNNERS

        runner = RUNNERS.build(cfg)

    checkpoint_ext = os.path.splitext(args.checkpoint)[-1]
    if checkpoint_ext == '.pth':
        runner.call_hook('before_run')
        runner.load_checkpoint(args.checkpoint, map_location=args.device)

        # TODO: Move metric hooks into config, only register here
        if args.task == 'mmcls':
            from mmengine import dump as mmdump
            from mmengine.hooks import Hook

            class SaveMetricHook(Hook):
                def after_test_epoch(self, _, metrics=None):
                    if metrics is not None:
                        print(metrics)
                        mmdump(metrics, args.dump)

            runner.register_hook(SaveMetricHook(), 'LOWEST')

        elif args.task == 'mmdet':
            from mmdet.utils import setup_cache_size_limit_of_dynamo

            setup_cache_size_limit_of_dynamo()

            if args.dump is not None:
                from mmdet.evaluation import DumpDetResults

                runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=args.dump))

        elif args.task == 'mmpose':
            if args.dump is not None:
                from mmengine import dump as mmdump
                from mmengine.hooks import Hook

                class SaveMetricHook(Hook):
                    def after_test_epoch(self, _, metrics=None):
                        if metrics is not None:
                            print(metrics)
                            mmdump(metrics, args.dump)

                runner.register_hook(SaveMetricHook(), 'LOWEST')

    elif checkpoint_ext in {'.tflite', '.onnx', '.param', '.bin'}:
        from sscma.utils import Infernce

        # TODO: Support inference '.tflite', '.onnx' model on different devices
        # TODO: Support MMEngine metric hooks
        # TODO: Support '.pickel' dump
        runner = Infernce(
            args.checkpoint,
            dataloader=runner.val_dataloader,
            cfg=cfg,
            runner=runner,
            dump=args.dump,
            source=args.source,
            task=str(args.task).replace('mm', ''),
            show=args.show,
            save_dir=args.out_dir,
        )

    runner.test()


if __name__ == '__main__':
    main()
