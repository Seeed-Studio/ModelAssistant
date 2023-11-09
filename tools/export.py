import argparse
import os
import tempfile

import torch
from tqdm import tqdm

# TODO: Move to config file
import sscma.datasets  # noqa
import sscma.engine  # noqa
import sscma.evaluation  # noqa
import sscma.models  # noqa
import sscma.visualization  # noqa
from sscma.utils.check import check_lib


def parse_args():
    from mmengine.config import DictAction

    parser = argparse.ArgumentParser(description='Convert and export PyTorch model to TFLite or ONNX models')

    # common configs
    parser.add_argument('config', type=str, help='the model config file path')
    parser.add_argument('checkpoint', type=str, help='the PyTorch checkpoint file path')
    parser.add_argument(
        '--targets',
        type=str,
        nargs='+',
        default=['tflite', 'onnx', 'pnnx', 'vela'],
        help='the target type of model(s) to export e.g. tflite onnx',
    )
    parser.add_argument(
        '--precisions',
        type=str,
        nargs='+',
        default=['int8', 'float32'],
        help="the precisions exported model, e.g. 'int8', 'uint8', 'int16', 'float16' and 'float32'",
    )
    parser.add_argument(
        '--work_dir',
        '--work-dir',
        type=str,
        default=None,
        help='the directory to save logs and models',
    )
    parser.add_argument(
        '--output_stem',
        '--output-stem',
        type=str,
        default=None,
        help='the stem of output file name (with path)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='the device used for convert & export',
    )
    parser.add_argument(
        '--input_shape',
        '--input-shape',
        type=int,
        nargs='+',
        default=None,
        help='the shape of input data, e.g. 1 3 224 224',
    )
    parser.add_argument(
        '--input_type',
        '--input-type',
        type=str,
        default='image',
        choices=['audio', 'image', 'sensor'],
        help='the type of input data',
    )
    parser.add_argument(
        '--cfg_options',
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help="override some settings in the used config, the key-value pair in 'xxx=yyy' format will be merged into config file",
    )
    parser.add_argument(
        '--vela',
        nargs='+',
        action=DictAction,
        help='Parameters required for exporting vela model, need to correspond to vela command line parameters.',
    )
    parser.add_argument(
        '--simplify',
        type=int,
        default=5,
        help='the level of graph simplification, 0 means disable, max: 5',
    )

    # ONNX specific
    parser.add_argument(
        '--opset_version',
        '--opset-version',
        type=int,
        default=11,
        help='ONNX: operator set version of exported model',
    )
    parser.add_argument(
        '--dynamic_export',
        '--dynamic-export',
        action='store_true',
        default=False,
        help='ONNX: export with a dynamic input shape',
    )

    # TFLite specific
    parser.add_argument(
        '--algorithm',
        type=str,
        default='l2',
        choices=['l2', 'kl'],
        help='TFLite: conversion algorithm',
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='fbgemm',
        choices=['qnnpack', 'fbgemm'],
        help='TFLite: converter backend',
    )
    parser.add_argument(
        '--calibration_epochs',
        '--calibration-epochs',
        type=int,
        default=100,
        help='TFLite: max epoches for quantization calibration',
    )
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[0.0],
        help='TFLite: mean for model input (quantization), range: [0, 1], applied to all channels, using the average if multiple values are provided',
    )
    parser.add_argument(
        '--mean_and_std',
        '--mean-and-std',
        type=str,
        nargs='+',
        default='[((0.0,), (1.0,))]',
        help='TFLite: mean and std for model input(s), default: [((0.0,), (1.0,))], calculated on normalized input(s), applied to all channel(s), using the average if multiple values are provided',
    )

    return parser.parse_args()


def verify_args(args):
    assert os.path.splitext(args.config)[-1] == '.py', "The config file name should be ended with a '.py' extension"
    assert os.path.exists(args.config), 'The config file does not exist'
    assert (
        os.path.splitext(args.checkpoint)[-1] == '.pth'
    ), "The chackpoint model should be a PyTorch model with '.pth' extension"
    assert os.path.exists(args.checkpoint), 'The chackpoint model does not exist'
    assert {str(t).lower() for t in args.targets}.issubset(
        {'tflite', 'onnx', 'pnnx', 'vela'}
    ), 'Supported in target type(s): onnx, tflite'
    assert {str(p).lower() for p in args.precisions}.issubset(
        {'int8', 'uint8', 'int16', 'float16', 'float32'}
    ), "Supported export precision(s): 'int8', 'uint8', 'int16', 'float16' and 'float32'"
    assert args.simplify in range(0, 5 + 1), 'Simplify level should be in [0, 5]'
    assert args.mean_and_std is not None or '', 'The mean and std value(s) for model input should be provided'

    return args


def build_config(args):
    from mmengine.config import Config

    from sscma.utils import load_config

    args.targets = [str(target).lower() for target in args.targets]
    args.precisions = [str(precision).lower() for precision in args.precisions]

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_data = load_config(args.config, folder=tmp_dir, cfg_options=args.cfg_options)
        cfg = Config.fromfile(cfg_data)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        args.work_dir = cfg.work_dir = os.path.join('work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    else:
        args.work_dir = cfg.work_dir

    if args.device.startswith('cuda'):
        args.device = args.device if torch.cuda.is_available() else 'cpu'

    cfg.val_dataloader['batch_size'] = 1
    cfg.val_dataloader['num_workers'] = 1

    if 'batch_shapes_cfg' in cfg.val_dataloader.dataset:
        cfg.val_dataloader.dataset.batch_shapes_cfg = None

    if args.input_shape is None:
        try:
            if 'imgsz' in cfg:
                args.input_shape = [1, 1 if cfg.get('gray', False) else 3, *cfg.imgsz]
            elif 'width' in cfg and 'height' in cfg:
                args.input_shape = [
                    1,
                    1 if cfg.get('gray', False) else 3,
                    cfg.width,
                    cfg.height,
                ]
        except Exception as exc:
            raise ValueError('Please specify the input shape') from exc
        print(
            "Using automatically generated input shape (from config '{}'): {}".format(
                os.path.basename(args.config), args.input_shape
            )
        )

    args.mean_and_std = [t for t in eval(args.mean_and_std)]
    for means, stds in args.mean_and_std:
        assert len(means) == len(stds), 'The mean and std values should be twin'
        assert all([0.0 <= m <= 1.0 for m in means]), 'Mean for model input should be in [0.0, 1.0]'

    return args, cfg


def calibrate(ptq_model, context, means_and_stds):
    # TODO: Support multiple inputs
    # TODO: Support handle 'audio', 'sensor' inputs
    ptq_model.to(device=context.device)
    ptq_model.eval()
    context.iteration = 0
    epoch = min(len(context.val_loader), context.max_iteration)
    with torch.no_grad(), tqdm(total=epoch, ncols=50) as pbar:
        for i, data in enumerate(context.val_loader):
            if context.max_iteration is not None and i >= context.max_iteration:
                break
            inputs = data['inputs']
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            assert isinstance(inputs, torch.Tensor), 'The input should be a tensor'
            if inputs.dtype != torch.float32:
                mean, std = means_and_stds[0]
                inputs = (inputs - mean) / std
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(0)
            inputs = inputs.to(device=context.device)

            ptq_model(inputs)

            pbar.update(1)
            context.iteration += 1


def get_exported_file_name_from_precision(args, precision, ext: str = '') -> str:
    return os.path.join(
        os.path.abspath(os.path.dirname(args.checkpoint)),
        os.path.splitext(os.path.basename(args.checkpoint if args.output_stem is None else args.output_stem))[0]
        + '_'
        + precision
        + ext,
    )


def export_pnnx(args, model):
    import sys
    import pnnx
    from pnnx.wrapper import convert_inputshape_to_cmd
    import os.path as osp

    model.eval()
    pnnx_bin_path = osp.join(osp.dirname(pnnx.__file__), 'bin')
    bin_list = os.listdir(pnnx_bin_path)
    x = torch.randn(args.input_shape)

    cmd = ""
    dir_dict = {}
    for dir_name in bin_list:
        if 'ubuntu' in dir_name:
            dir_dict["linux"] = dir_name
        elif 'windows' in dir_name:
            dir_dict['win'] = dir_name
        elif 'macos' in dir_name:
            dir_dict['darwin'] = dir_name
    if sys.platform.startswith('linux'):
        cmd += os.path.join(pnnx_bin_path, dir_dict["linux"], "pnnx ")
    elif sys.platform.startswith('win'):
        cmd += os.path.join(pnnx_bin_path, dir_dict["win"], "pnnx.exe ")
    elif sys.platform.startswith('darwin'):
        cmd += os.path.join(pnnx_bin_path, dir_dict["darwin"], "pnnx ")
    cmd += 'model.pt '
    cmd += convert_inputshape_to_cmd(x)
    cmd += " pnnxparam=" + get_exported_file_name_from_precision(args, 'float', '.pnnx.param')
    cmd += " pnnxbin=" + get_exported_file_name_from_precision(args, 'float', '.pnnx.bin')
    cmd += " pnnxpy=" + get_exported_file_name_from_precision(args, 'float', '.pnnx.py')
    cmd += " pnnxonnx=" + get_exported_file_name_from_precision(args, 'float', '.pnnx.onnx')
    cmd += " ncnnparam=" + get_exported_file_name_from_precision(args, 'float', '.ncnn.param')
    cmd += " ncnnbin=" + get_exported_file_name_from_precision(args, 'float', '.ncnn.bin')
    cmd += " ncnnpy=" + get_exported_file_name_from_precision(args, 'float', '.ncnn.py')

    trace_model = torch.jit.trace(model, x)
    trace_model.save('model.pt')
    os.system(cmd)


def export_tflite(args, model, loader):
    from tinynn.converter import TFLiteConverter
    from tinynn.graph.quantization.quantizer import PostQuantizer
    from tinynn.graph.tracer import model_tracer
    from tinynn.util.train_util import DLContext

    context = DLContext()
    context.device = args.device
    context.val_loader = loader
    context.max_iteration = args.calibration_epochs

    # TODO: Support multiple inputs
    with torch.no_grad():
        model.eval()
        dummy_input = torch.randn(*args.input_shape, requires_grad=False).to(device=context.device)

    for precision in args.precisions:
        if precision not in ['int8', 'uint8', 'int16', 'float32']:
            print('TFLite: Ignoring unsupported precision: {}'.format(precision))
            continue

        tflite_file = get_exported_file_name_from_precision(args, precision, '.tflite')
        if precision in ['int8', 'uint8', 'int16']:
            # TODO: Support handle 'audio', 'sensor' inputs
            if args.input_type == 'image':
                means_and_stds = [
                    (torch.mean(torch.tensor(ms)).item() * 255.0, torch.mean(torch.tensor(ss)).item() * 255.0)
                    for ms, ss in args.mean_and_std
                ]
            else:
                raise NotImplementedError
            with model_tracer():
                quantizer = PostQuantizer(
                    model,
                    dummy_input,
                    work_dir=args.work_dir,
                    config={
                        'asymmetric': True,
                        'set_quantizable_op_stats': True,
                        'per_tensor': False,
                        'algorithm': args.algorithm,
                        'backend': args.backend,
                        'quantized_input_stats': means_and_stds,
                    },
                )
                ptq_model = quantizer.quantize()
                ptq_model.to(device=context.device)

            calibrate(ptq_model, context, means_and_stds)

            with torch.no_grad():
                ptq_model.eval()
                ptq_model = quantizer.convert(ptq_model)
                torch.backends.quantized.engine = quantizer.backend
                converter = TFLiteConverter(
                    ptq_model,
                    dummy_input,
                    optimize=args.simplify,
                    quantize_target_type=precision,
                    fuse_quant_dequant=True,
                    rewrite_quantizable=True,
                    tflite_micro_rewrite=True,
                    tflite_path=tflite_file,
                )
        else:
            with torch.no_grad():
                converter = TFLiteConverter(
                    model,
                    dummy_input,
                    optimize=args.simplify,
                    tflite_path=tflite_file,
                )

        try:
            converter.convert()
        except Exception as exp:
            raise RuntimeError('TFLite: Failed exporting the model') from exp
        else:
            if 'vela' in args.targets:
                export_vela(args, tflite_file)

        print('TFLite: Successfully export model: {}'.format(tflite_file))


def export_onnx(args, model):
    with torch.no_grad():
        model.eval()
        dummy_input = torch.randn(*args.input_shape, requires_grad=False).to(device=args.device)

    for precision in args.precisions:
        if precision not in ['float32']:
            print('ONNX: Ignoring unsupported precision: {}'.format(precision))
            continue

        onnx_file = get_exported_file_name_from_precision(args, precision, '.onnx')
        if args.dynamic_export:
            # TODO: Implement dynamic export
            raise NotImplementedError
        else:
            dynamic_axes = {}

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_file,
                input_names=['input'],
                output_names=['output'],
                export_params=True,
                keep_initializers_as_inputs=True,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                verbose=True,
                opset_version=args.opset_version,
            )

        if args.simplify > 0:
            import onnx
            import onnxsim

            model_simp, check = onnxsim.simplify(onnx_file)
            if check:
                onnx.save(model_simp, onnx_file)
            else:
                print("ONNX: Failed to simplify the model: '{}', revert to the original".format(onnx_file))
        print('ONNX: Successfully export model: {}'.format(onnx_file))


def export_vela(args, model):
    if check_lib('ethos-u-vela'):
        from ethosu.vela.vela import main as vela_main
    else:
        raise ImportError(
            'An error occurred while importing "ethosu", please check if "ethosu" is already installed,',
            'or run "pip install ethos-u-vela" and try again.',
        )

    vela_args = [model, '--output-dir', os.path.dirname(model)]
    if args.vela is not None:
        for key, value in args.vela.items():
            vela_args.append('--' + key)
            vela_args.append(value)
    vela_main(vela_args)


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

    runner.call_hook('before_run')
    runner.load_checkpoint(args.checkpoint, map_location=args.device)

    model = runner.model.to(device=args.device)
    loader = runner.val_dataloader

    for target in args.targets:
        if target == 'tflite':
            export_tflite(args, model, loader)
        elif target == 'onnx':
            export_onnx(args, model)
        elif target == 'pnnx':
            export_pnnx(args, model)


if __name__ == '__main__':
    main()
