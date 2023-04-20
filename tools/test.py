import os
import sys
import warnings
import argparse
import warnings
import os.path as osp
from pathlib import Path

import mmcv
import torch
from mmcv.cnn import fuse_conv_bn
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from tools.utils.inference import Inter, pfld_inference, audio_inference, show_point, fomo_inference
from edgelab.engine.apis.mmdet import single_gpu_test_fomo, single_gpu_test_mmcls, multi_gpu_test
from edgelab.engine.utils.helper_funcs import check_type
from tools.utils.config import load_config

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model

# Not display redundant warnning.
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Get work dir.
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


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
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('type', help='Choose training type')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--audio', action='store_true', help='Choose audio dataset load code if given')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--data', help='point data root manually')
    parser.add_argument('--no-show',
                        action='store_true',
                        help='Whether to display the results after inference')
    parser.add_argument('--save-dir',
                        default=None,
                        help='Folder to save results...')
    parser.add_argument('--work-dir',
                        help='the dir to save evaluation results')

    parser.add_argument('--gpu-id',
                        type=int,
                        default=0,
                        help='id of gpu to use '
                        '(only applicable to non-distributed testing)')
    parser.add_argument('--eval',
                        default=None,
                        nargs='+',
                        help='evaluation metric, which depends on the dataset,'
                        ' e.g., "mAP" for MSCOCO')
    parser.add_argument('--gpu-collect',
                        action='store_true',
                        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    # cfg = Config.fromfile(args.config)
    config_data = load_config(args.config, args.cfg_options)
    cfg = Config.fromstring(config_data,
                            file_format=osp.splitext(args.config)[-1])

     # load bulid function depends on type
    setup_multi_processes, build_model, build_dataset, build_dataloader = check_type(args.type)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.data is not None:
        args.data = os.path.abspath(args.data)
        cfg.data_root = args.data
        cfg.data.train.data_root = args.data
        cfg.data.val.data_root = args.data
        cfg.data.test.data_root = args.data

    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=1),
        **cfg.data.get('test_dataloader', {})
    }

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    os.chdir(ROOT)  # build_dataset function will change the root dir.
    data_loader = [build_dataloader(ds, **test_loader_cfg) for ds in dataset]
    # data_loader = dataset

    # build the model and load checkpoint
    model, pt = load_model(cfg, args.checkpoint, build_model, args.fuse_conv_bn)

    if pt:
        if not distributed:
            model = MMDataParallel(model, device_ids=[args.gpu_id])
            # outputs = single_gpu_test(model, data_loader)
            if (dataset.__class__.__name__ == "FomoDatasets"):
                outputs = single_gpu_test_fomo(model, data_loader)
            else:
                outputs = single_gpu_test_mmcls(model, data_loader, args.audio)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)
    elif args.audio:
        outputs = audio_inference(model, data_loader)
    elif (dataset.__class__.__name__ == "MeterData"):
        outputs = pfld_inference(model, data_loader)
    elif (dataset.__class__.__name__ == "FomoDatasets"):
        outputs = fomo_inference(model, data_loader)

    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))

    if (not args.no_show or args.save_dir) and not args.audio:
        for out in outputs:
            if pt:
                model.module.show_result(
                    out['image_file'],
                    out['result'][0],
                    show=False if args.no_show else True,
                    win_name='test',
                    save_path=args.save_dir if args.save_dir else None,
                    **out)
            else:
                show_point(out['pred'],
                           out['image_file'],
                           save_path=args.save_dir if args.save_dir else None,
                           not_show=args.no_show)

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        results = dataset.evaluate(outputs, **eval_config)
        print('\n')
        print('=' * 30)
        for k, v in sorted(results.items()):
            print(f'{k}: {v}')
        print('=' * 30)


if __name__ == '__main__':
    main()