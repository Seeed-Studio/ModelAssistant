import argparse
import os
import os.path as osp
import tempfile as tf
from typing import Optional, Tuple

import numpy as np
import onnxruntime as rt
import torch
import onnx
from onnx.checker import check_model

from mmengine.config import Config
from mmengine.runner import Runner
from tools.utils.config import load_config
import edgelab.models
import edgelab.datasets
import edgelab.evaluation

torch.manual_seed(3)


def parse_normalize_cfg(test_pipeline):
    transforms = None
    for pipeline in test_pipeline:
        if 'transforms' in pipeline:
            transforms = pipeline['transforms']
            break
    assert transforms is not None, 'Failed to find `transforms`'
    norm_config_li = [_ for _ in transforms if _['type'] == 'Normalize']
    assert len(norm_config_li) == 1, '`norm_config` should only have one'
    norm_config = norm_config_li[0]
    return norm_config


def _demo_mm_inputs(input_shape, num_classes):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions
        num_classes (int):
            number of semantic classes
    """
    # (N, C, H, W) = input_shape
    # input_shape = (1, 1, 16384)
    rng = np.random.RandomState(0)
    imgs = rng.rand(*input_shape)
    gt_labels = rng.randint(low=0, high=num_classes,
                            size=(1, 1)).astype(np.uint8)
    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'gt_labels': torch.LongTensor(gt_labels),
    }
    return mm_inputs


def pytorch2onnx(model: torch.nn.Module,
                 input_shape: Tuple[int, ...],
                 input_img: Optional[str] = None,
                 normalize: Optional[Tuple[float, ...]] = None,
                 opset_version: int = 9,
                 dynamic_export: bool = False,
                 show: bool = False,
                 output_file='tmp.onnx',
                 do_simplify: bool = False,
                 verify: bool = False):
    """Export Pytorch model to ONNX model and verify the outputs are same
    between Pytorch and ONNX.

    Args:
        model (nn.Module): Pytorch model we want to export.
        input_shape (tuple): Use this input shape to construct
            the corresponding dummy input and execute the model.
        opset_version (int): The onnx op version. Default: 11.
        show (bool): Whether print the computation graph. Default: False.
        output_file (string): The path to where we store the output ONNX model.
            Default: `tmp.onnx`.
        verify (bool): Whether compare the outputs between Pytorch and ONNX.
            Default: False.
    """
    model.cpu().eval()

    input_img = torch.randn(size=input_shape)

    # support dynamic shape export
    if dynamic_export:
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width'
            },
            'output': {
                0: 'batch'
            }
        } if len(input_shape) == 3 else {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'output': {
                0: 'batch'
            }
        }
    else:
        dynamic_axes = {}

    # export onnx
    with torch.no_grad():
        torch.onnx.export(model,
                          input_img,
                          output_file,
                          input_names=['input'],
                          output_names=['output'],
                          export_params=True,
                          keep_initializers_as_inputs=True,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          verbose=show,
                          opset_version=opset_version)
        print(f'Successfully exported ONNX model: {output_file}')

    if do_simplify:
        import onnxsim

        model_opt, check_ok = onnxsim.simplify(output_file)
        if check_ok:
            onnx.save(model_opt, output_file)
            print(f'Successfully simplified ONNX model: {output_file}')
        else:
            print('Failed to simplify ONNX model.')
    if verify:  # TODO
        # check by onnx
        onnx_model = onnx.load(output_file)
        check_model(onnx_model)

        # test the dynamic model
        if dynamic_export:
            dynamic_test_inputs = _demo_mm_inputs(
                (input_shape[0], input_shape[1], input_shape[2] * 2,
                 input_shape[3] * 2), model.head.num_classes)
            imgs = dynamic_test_inputs.pop('imgs')
            img_list = [img[None, :] for img in imgs]

        # check the numerical value
        # get pytorch output
        pytorch_result = model(img_list, img_metas={}, return_loss=False)[0]

        # get onnx output
        input_all = [node.name for node in onnx_model.graph.input]
        input_initializer = [
            node.name for node in onnx_model.graph.initializer
        ]

        net_feed_input = list(set(input_all) - set(input_initializer))
        assert (len(net_feed_input) == 1)
        sess = rt.InferenceSession(output_file)
        onnx_result = sess.run(
            None, {net_feed_input[0]: img_list[0].detach().numpy()})[0]
        if not np.allclose(pytorch_result, onnx_result):
            raise ValueError(
                'The outputs are different between Pytorch and ONNX')
        print('The outputs are same between Pytorch and ONNX')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Pytorch model to ONNX')
    parser.add_argument('task',
                        type=str,
                        default='cls',
                        choices=['cls', 'det', 'pose'],
                        help='The task type of the exported model')
    parser.add_argument(
        'config',
        type=str,
        default='./configs/audio_classify/ali_classiyf_small_8k_8192.py',
        help='test config file path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file')

    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--verify',
                        action='store_true',
                        default=False,
                        help='verify the onnx model')
    parser.add_argument('--output-file',
                        type=str,
                        help='Exported onnx file name')
    parser.add_argument('--opset-version',
                        type=int,
                        default=11,
                        help='Exported version of onnx operator set')
    parser.add_argument('--simplify',
                        action='store_true',
                        default=False,
                        help='Whether to simplify onnx model.')
    parser.add_argument('--shape',
                        type=int,
                        nargs='+',
                        default=[112],
                        help='input data size')
    parser.add_argument('--audio',
                        action='store_true',
                        default=False,
                        help='Whether the input data is audio data')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        default=False,
        help='Whether to export ONNX with dynamic input shape. \
            Defaults to False.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    shape = args.shape
    audio = args.audio
    normalize_cfg = None
    assert not (len(shape) == 2 and audio), 'When the model input data is audio data, its input shape should be BXCXW, ' \
                                            'but when receiving data from BXCXHXW, please check whether the input data shape is correct'

    if audio:
        input_shape = (1, 1, shape[0])
    elif len(shape) == 1:
        input_shape = (1, 3, shape[0], shape[0])
    elif len(shape) == 2:
        input_shape = (
            1,
            3,
        ) + tuple(shape)
    else:
        raise ValueError('invalid input shape')

    # load config
    tmp_folder = tf.TemporaryDirectory()
    # Modify and create temporary configuration files
    config_data = load_config(args.config, folder=tmp_folder.name)
    # load temporary configuration files
    cfg = Config.fromfile(config_data)
    tmp_folder.cleanup()
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])

    runner = Runner.from_cfg(cfg=cfg)
    runner.call_hook('before_run')

    # build the model
    # if args.task == 'mmcls':
    #     model = build_classifier(cfg.model)
    # elif args.task == 'mmdet':
    #     model = build_detector(cfg.model)
    #     normalize_cfg = parse_normalize_cfg(cfg.test_pipeline)
    #     args.opset_version = 11
    # elif args.task == 'mmpose':
    #     model = build_posenet(cfg.model)

    # load checkpoint
    if args.checkpoint:
        runner.load_checkpoint(filename=args.checkpoint)

    if not args.input_img:
        args.input_img = os.path.join(os.path.dirname(__file__),
                                      '../demo/demo.jpg')

    if args.output_file:
        output_file = os.path.join(os.path.dirname(args.checkpoint),
                                   args.output_file)
    else:
        output_file = osp.abspath(args.checkpoint)
        bn = osp.basename(output_file)
        dn = osp.dirname(output_file)
        output_file = osp.join(dn, bn.replace('.pth', '.onnx'))

    # convert model to onnx file
    pytorch2onnx(runner.model,
                 input_shape,
                 normalize=normalize_cfg,
                 input_img=args.input_img,
                 opset_version=args.opset_version,
                 show=args.show,
                 dynamic_export=args.dynamic_export,
                 output_file=output_file,
                 do_simplify=args.simplify,
                 verify=args.verify)


if __name__ == '__main__':
    main()