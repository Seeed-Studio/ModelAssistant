from argparse import ArgumentParser

from mmengine.registry import DefaultScope

from sscma.infer import CustomInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-file', default='result', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):
    DefaultScope.get_instance(
        name='sscma', scope_name='sscma')
    inferencer = CustomInferencer(
        args.config, args.checkpoint, save_path=args.out_file)
    inferencer(args.img, vis_thresh=0.8)


if __name__ == '__main__':
    args = parse_args()
    main(args)
