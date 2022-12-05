import argparse

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--shape',
                        type=int,
                        nargs='+',
                        default=[320, 320],
                        help='input image size')
    parser.add_argument('--task',
                        default='mmcls',
                        help='The task type to which the model belongs')
    parser.add_argument('--audio',
                        action='store_true',
                        help='input data is audio')
    parser.add_argument('--channle',
                        type=int,
                        default=3,
                        help='Number of channels for input data')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        if args.audio:
            input_shape = (1, args.shape[0])
        else:
            input_shape = (args.channle, args.shape[0], args.shape[0])

    elif len(args.shape) == 2:
        input_shape = (args.channle, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.task == 'mmcls':
        from mmcls.models import build_classifier

        model = build_classifier(cfg.model)
    elif args.task == 'mmdet':
        from mmdet.models import build_detector

        model = build_detector(cfg.model)
    elif args.task == 'mmpose':
        from mmpose.models import build_posenet

        model = build_posenet(cfg.model)
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model,
                                              input_shape,
                                              as_strings=True)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()