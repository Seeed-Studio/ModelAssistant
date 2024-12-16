import argparse
import os
import sys
import os.path as osp
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.util.train_util import AverageMeter
from tinynn.graph.tracer import model_tracer
from tinynn.graph.quantization.algorithm.cross_layer_equalization import (
    cross_layer_equalize,
)
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize
from tinynn.converter import TFLiteConverter
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.device import get_device

from mmengine import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description="quantizer a model")
    parser.add_argument("config", help="quantization config file path")
    parser.add_argument("model", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--test", action="store_true", help="Whether to evaluate inference results"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=dict(epochs=5),
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def calibrate(model, runner, max_iteration=None, device="cpu"):
    """Calibrates the fake-quantized model

    Args:
        model: The model to be validated
    """

    model.eval()
    model = model.to(device)
    runner.model.data_preprocessor.to(device)

    avg_batch_time = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, data_batch in enumerate(
            tqdm(runner.train_dataloader, desc="calibrate model")
        ):
            if max_iteration is not None and idx >= max_iteration:
                break
            datas = runner.model.data_preprocessor(data_batch, False)

            for data in datas["inputs"].split(1, 0):
                out = model(data)

            # measure elapsed time
            avg_batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                print(
                    f"Calibrate: [{idx}/{len(runner.train_dataloader)}]\tTime {avg_batch_time.avg:.5f}\t"
                )


def plot_output_channel(X):
    X = X.detach().cpu()
    C = X.shape[1]
    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    ranges = torch.zeros((X.size(1), 2))  # Two columns for min and max
    for i in range(X.size(1)):  # Iterate over channels
        channel_data = X[:, i, :, :].flatten()  # Flatten the spatial dimensions
        ranges[i, 0] = torch.min(channel_data)  # Min value for this channel
        ranges[i, 1] = torch.max(channel_data)  # Max value for this channel

    # Convert the ranges to a format suitable for box plot
    ranges = ranges.numpy()

    # Create the box plot
    ax1.boxplot(ranges.transpose(), positions=range(1, C + 1), showfliers=False)

    ax1.set_title("Range of values for each output channel")

    ax2.hist(X.flatten().detach().numpy(), bins=100)
    # Plot data on the second subplot
    ax2.set_title("Distribution")
    # show the plot
    # plt.show()


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config, modified_constant=args.cfg_options)
    cfg.launcher = args.launcher

    # multiprocessing.set_start_method("spawn")
    # # onnxruntime does not support fork method in multiprocessing
    # cfg.env_cfg.mp_cfg.mp_start_method = "spawn"

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    # load hook config
    cfg.custom_hooks = [
        dict(
            type="QuantizerSwitchHook",
            freeze_quantizer_epoch=cfg.epochs if hasattr(cfg, "epochs") else 5 // 3,
            freeze_bn_epoch=cfg.epochs if hasattr(cfg, "epochs") else 5 // 3 * 2,
        ),
    ]

    # remove amp
    if (
        hasattr(cfg.optim_wrapper, "type")
        and cfg.optim_wrapper.type.__name__ == "AmpOptimWrapper"
    ):
        cfg.optim_wrapper.type = "OptimWrapper"

    cfg.load_from = args.model

    imgsz = cfg.get("imgsz")

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    # test for original pytorch fp32 model
    if args.test:
        runner.test()

    model = runner.model

    model = cross_layer_equalize(
        model, torch.randn(1, 3, imgsz[0], imgsz[1]), get_device()
    )

    # init quant model
    with model_tracer():
        model.to("cpu")
        dummy_input = torch.randn(1, 3, imgsz[0], imgsz[1])
        # model_copy = cross_layer_equalize(model_copy, dummy_input, get_device())
        quantizer = QATQuantizer(
            model,
            dummy_input,
            work_dir="out",
            config={
                "asymmetric": True,
                "force_overwrite": True,
                "per_tensor": False,
                "disable_requantization_for_cat": True,
                "override_qconfig_func": set_ptq_fake_quantize,
            },
        )
        ptq_model = quantizer.quantize()

        quantizer.optimize_conv_bn_fusion(ptq_model)

    q_model = MODELS.build(cfg.quantizer_config)
    q_model.set_model(ptq_model)

    q_model.to(get_device())
    runner.model = q_model

    # test initial quantized model
    if args.test:
        runner.test()

    # qat train
    runner.train()

    # export to tflite
    with torch.no_grad():
        ptq_model.eval()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        ptq_model.cpu()
        ptq_model = quantizer.convert(ptq_model, backend="qnnpack")

        tf_converter = TFLiteConverter(
            ptq_model,
            torch.randn(3, 3, imgsz[0], imgsz[1]),
            tflite_path="out/qat_model_test.tflite",
            fuse_quant_dequant=True,
            quantize_target_type="int8",
        )
        tf_converter.convert()


if __name__ == "__main__":
    main()
