"""
 python export.py --config configs/rtmdet_nano_8xb32_300e_coco.py --weights  work_dirs/rtmdet_nano_8xb32_300e_coco/epoch_300.pth --include onnx torchscript
todo
imsz ?
"""

import inspect
import argparse
import inspect
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

import sscma
from mmengine.config import Config
from mmengine.logging import print_log
from sscma.apis import init_detector

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # sscma root directory


def colorstr(*input):
    """
    Colors a string using ANSI escape codes, e.g., colorstr('blue', 'hello world').

    See https://en.wikipedia.org/wiki/ANSI_escape_code.
    """
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Logs the arguments of the calling function, with options to include the filename and function name."""
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    print_log(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))


def export_formats():
    r"""
    Returns a DataFrame of supported sscma model export formats and their properties.

    Returns:
        pandas.DataFrame: A DataFrame containing supported export formats and their properties. The DataFrame
        includes columns for format name, CLI argument suffix, file extension or directory name, and boolean flags
        indicating if the export format supports training and detection.

    Examples:
        ```python
        formats = export_formats()
        print(f"Supported export formats:\n{formats}")
        ```

    Notes:
        The DataFrame contains the following columns:
        - Format: The name of the model format (e.g., PyTorch, TorchScript, ONNX, etc.).
        - Include Argument: The argument to use with the export script to include this format.
        - File Suffix: File extension or directory name associated with the format.
        - Supports Training: Whether the format supports training.
        - Supports Detection: Whether the format supports detection.
    """
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def select_device(device="", batch_size=0, newline=True):
    """Selects computing device (CPU, CUDA GPU, MPS) for RTMDet model deployment, logging device info."""
    s = f"Seeed Studio SenseCraft Model Assistant:👨🏻‍💻👨🏻‍💻🦾🧠 sscma version: {sscma.version.__version__}, torch-{torch.__version__} "
    device = (
        str(device).strip().lower().replace("cuda:", "").replace("none", "")
    )  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            "-1"  # force torch.cuda.is_available() = False
        )
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            device  # set environment variable - must be before assert is_available()
        )
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(
            device.replace(",", "")
        ), f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = (
            device.split(",") if device else "0"
        )  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert (
                batch_size % n == 0
            ), f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif (
        mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available()
    ):  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    print_log(s)
    return arg


def make_divisible(x, divisor):
    """Adjusts `x` to be divisible by `divisor`, returning the nearest greater or equal value."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    """Adjusts image size to be divisible by stride `s`, supports int or list/tuple input, returns adjusted size."""
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print_log(
            f"WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}"
        )
    return new_size


def file_size(path):
    """Returns file or directory size in megabytes (MB) for a given path, where directories are recursively summed."""
    mb = 1 << 20  # bytes to MiB (1024 ** 2)
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / mb
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / mb
    else:
        return 0.0


def export_torchscript(model, im, file, optimize, prefix=colorstr("TorchScript:")):
    """
    Export a RTMDet model to the TorchScript format.

    Args:
        model (torch.nn.Module): The RTMDet model to be exported.
        im (torch.Tensor): Example input tensor to be used for tracing the TorchScript model.
        file (Path): File path where the exported TorchScript model will be saved.
        optimize (bool): If True, applies optimizations for mobile deployment.
        prefix (str): Optional prefix for log messages. Default is 'TorchScript:'.

    Returns:
        (str | None, torch.jit.ScriptModule | None): A tuple containing the file path of the exported model
            (as a string) and the TorchScript model (as a torch.jit.ScriptModule). If the export fails, both elements
            of the tuple will be None.

    Notes:
        - This function uses tracing to create the TorchScript model.
        - Metadata, including the input image shape, model stride, and class names, is saved in an extra file (`config.txt`)
          within the TorchScript model package.
        - For mobile optimization, refer to the PyTorch tutorial: https://pytorch.org/tutorials/recipes/mobile_interpreter.html

    """
    print_log(f"\n{prefix} starting export with torch {torch.__version__}...")
    f = file.with_suffix(".torchscript")

    ts = torch.jit.trace(model, im, strict=False)
    d = {"dataset": model.dataset_meta, "names": model.cfg.filename}
    extra_files = {"config.txt": json.dumps(d)}  # torch._C.ExtraFilesMap()
    if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
        optimize_for_mobile(ts)._save_for_lite_interpreter(
            str(f), _extra_files=extra_files
        )
    else:
        ts.save(str(f), _extra_files=extra_files)
    return f, None


def export_engine(
    model,
    im,
    file,
    half,
    dynamic,
    simplify,
    workspace=4,
    verbose=False,
    prefix=colorstr("TensorRT:"),
):
    # TODO: Add export_engine function
    pass


def export_openvino(
    file, metadata, half, int8, data=None, prefix=colorstr("OpenVINO:")
):
    # TODO: Add export_openvino function
    pass


def export_saved_model(
    model,
    im,
    file,
    dynamic,
    keras=False,
    prefix=colorstr("TensorFlow SavedModel:"),
):
    # TODO: Add export_saved_model function
    pass


def export_tflite(
    keras_model,
    im,
    file,
    int8,
    per_tensor,
    data,
    prefix=colorstr("TensorFlow Lite:"),
):
    # TODO: Add export_saved_model function
    pass


def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr("ONNX:")):
    """
    Export a RTMDet model to ONNX format with dynamic axes support and optional model simplification.

    Args:
        model (torch.nn.Module): The RTMDet model to be exported.
        im (torch.Tensor): A sample input tensor for model tracing, usually the shape is (1, 3, height, width).
        file (pathlib.Path | str): The output file path where the ONNX model will be saved.
        opset (int): The ONNX opset version to use for export.
        dynamic (bool): If True, enables dynamic axes for batch, height, and width dimensions.
        prefix (str): A prefix string for logging messages, defaults to 'ONNX:'.

    Returns:
        tuple[pathlib.Path | str, None]: The path to the saved ONNX model file and None (consistent with decorator).

    Raises:
        ImportError: If required libraries for export (e.g., 'onnx', 'onnx-simplifier') are not installed.
        AssertionError: If the simplification check fails.

    Notes:
        The required packages for this function can be installed via:
        ```
        pip install onnx
        ```
    """
    # install "onnx>=1.12.0"
    import onnx

    print_log(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = str(file.with_suffix(".onnx"))

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["images"],
        output_names=None,
        dynamic_axes=None,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {"dataset": model.dataset_meta, "names": model.cfg.filename}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)
    return f, model_onnx


def run(
    config=ROOT / "configs/rtmdet_nano_8xb32_300e_coco.py",
    weights=ROOT / "sscma.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    include=("torchscript", "onnx"),  # include formats
    half=False,  # FP16 half-precision export
    optimize=False,  # TorchScript: optimize for mobile
    per_tensor=False,  # TF: per-tensor quantization
    int8=False,  # TF INT8 quantization
    dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    simplify=False,  # ONNX: simplify model
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
):
    """
    Exports a RTMDet model to specified formats including ONNX, TensorRT, CoreML, and TensorFlow.
    Returns:
        None
    Notes:
        - Model export is based on the specified formats in the 'include' argument.
        - Be cautious of combinations where certain flags are mutually exclusive, such as `--half` and `--dynamic`.
    """
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()["Argument"][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(
        include
    ), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    jit, onnx, xml, engine, saved_model, pb, tflite = flags  # export booleans
    file = Path(weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert (
            device != "cpu"
        ), "--half only compatible with GPU export, i.e. use --device 0"
        assert (
            not dynamic
        ), "--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both"

    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    if "init_cfg" in config.model.backbone:
        config.model.backbone.init_cfg = None

    model = init_detector(config, weights, device=device, cfg_options={})

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert (
            device == "cpu"
        ), "--optimize not compatible with cuda devices, i.e. use --device cpu"

    # Input
    gs = 32  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(
        device
    )  # image size(1,3,320,192) BCHW iDetection

    for _ in range(2):
        model(im)  # dry runs
    if half:
        im, model = im.half(), model.half()  # to FP16
    metadata = {
        "dataset": model.dataset_meta,
        "names": model.cfg.filename,
    }  # model metadata
    print_log(
        f"\n{colorstr('PyTorch:')} starting from {file}  ({file_size(file):.1f} MB)"
    )

    # Exports
    f = [""] * len(fmts)  # exported filenames
    warnings.filterwarnings(
        action="ignore", category=torch.jit.TracerWarning
    )  # suppress TracerWarning
    if jit:  # TorchScript
        f[0], _ = export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT required before ONNX
        f[1], _ = export_engine(
            model, im, file, half, dynamic, simplify, workspace, verbose
        )
    if onnx or xml:  # OpenVINO requires ONNX
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify)
    if xml:  # OpenVINO
        f[3], _ = export_openvino(file, metadata, half, int8)

    if any((saved_model, tflite)):  # TensorFlow formats
        # TODO: Add export_saved_model function
        # TODO: Add export_tflite function
        pass
    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        h = "--half" if half else ""  # --half FP16 inference arg
        dir = Path("")
        all_s = (
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
        )
        for x in f:
            all_s += f"\nValidate:        python {dir / 'demo.py'} --weights {x} {h}"
        all_s += f"\nVisualize:       https://netron.app"
        print_log(all_s)
    return f  # return list of exported files/dirs


def parse_opt(known=False):
    """
    Parse command-line options for RTMDet model export configurations.

    Args:
        known (bool): If True, uses `argparse.ArgumentParser.parse_known_args`; otherwise, uses `argparse.ArgumentParser.parse_args`.
                      Default is False.

    Returns:
        argparse.Namespace: Object containing parsed command-line arguments.

    Example:
        ```python
        opts = parse_opt()
        print(opts.data)
        print(opts.weights)
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "sscma.pt",
        help="model.pt path(s)",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image (h, w)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--half", action="store_true", help="FP16 half-precision export"
    )
    parser.add_argument(
        "--optimize", action="store_true", help="TorchScript: optimize for mobile"
    )
    parser.add_argument(
        "--int8", action="store_true", help="TF/OpenVINO INT8 quantization"
    )
    parser.add_argument(
        "--per-tensor",
        action="store_true",
        help="TF per-tensor quantization",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes"
    )
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument(
        "--workspace", type=int, default=4, help="TensorRT: workspace size (GB)"
    )

    parser.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help="torchscript, onnx, openvino, engine,  saved_model, tflite",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Run(**vars(opt))  # Execute the run function with parsed options."""
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
