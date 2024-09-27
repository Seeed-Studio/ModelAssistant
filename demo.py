import os

from argparse import ArgumentParser
from pathlib import Path


import torch

from mmengine import MODELS
from mmengine.config import Config
from mmengine.utils import ProgressBar, path, scandir
from mmengine.dataset import Compose
from mmengine.dataset import default_collate


from mmengine.registry import VISUALIZERS, DATASETS

from sscma.utils import simplecv_imread
from sscma.utils.colorspace import simplecv_imconvert
from sscma.deploy.backend import OnnxInfer, TorchScriptInfer
from sscma.deploy.utils import get_file_list, model_type


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("img", help="Image path, include image file, dir and URL.")
    parser.add_argument("config", help="Config file")
    parser.add_argument("model", help="model file")
    parser.add_argument("--out-dir", default="./output", help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--show", action="store_true", help="Show the detection results"
    )
    parser.add_argument(
        "--score-thr", type=float, default=0.3, help="Bbox score threshold"
    )
    parser.add_argument(
        "--class-name", nargs="+", type=str, help="Only Save those classes if set"
    )
    parser.add_argument(
        "--to-labelme", action="store_true", help="Output labelme style label file"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = args.config

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )

    # get file list
    files, source_type = get_file_list(args.img)

    # build model
    model_infer = MODELS.build(config.deploy)

    # init visualizer
    visualizer = VISUALIZERS.build(config.visualizer)
    dataset = DATASETS.build(config.val_dataloader.dataset)
    visualizer.dataset_meta = dataset.metainfo

    # select backend
    backend = model_type(args.model)
    if backend[1]:  # torchscript
        infer_torchscript_model = TorchScriptInfer(args.model, args.device)
        model_infer.set_infer(infer_torchscript_model, config)
    elif backend[2]:  # onnx
        infer_onnx_model = OnnxInfer(args.model, args.device)
        model_infer.set_infer(infer_onnx_model, config)

    # init test pipeline
    test_pipeline = Compose(config.test_pipeline)

    # init data preprocessor
    data_preprocessor = MODELS.build(config.model.data_preprocessor)

    progress_bar = ProgressBar(len(files))
    for file in files:
        with torch.no_grad():
            data = test_pipeline(dict(img_path=file, img_id=0))
            data = default_collate([data])
            data = data_preprocessor(data, False)
            result = model_infer.forward(
                inputs=data["inputs"], data_samples=data["data_samples"], mode="predict"
            )

        img = simplecv_imread(file)
        img = simplecv_imconvert(img, "bgr", "rgb")

        if source_type["is_dir"]:
            filename = os.path.relpath(file, args.img).replace("/", "_")
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)

        progress_bar.update()

        # TODO: Add support for labelme output
        # if args.to_labelme:
        #     # save result to labelme files
        #     out_file = out_file.replace(os.path.splitext(out_file)[-1], ".json")
        #     to_label_format(pred_instances, result.metainfo, out_file, args.class_name)
        #     continue

        visualizer.add_datasample(
            filename,
            img,
            data_sample=result[0],
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr,
        )


if __name__ == "__main__":
    main()
