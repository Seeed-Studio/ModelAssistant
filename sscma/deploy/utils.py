import os
import pandas as pd
import urllib

from pathlib import Path
from mmengine.utils import scandir

import torch

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def export_formats():
    r"""
    Returns a DataFrame of supported YOLOv5 model export formats and their properties.

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
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def check_suffix(file="sscma.pt", suffix=(".pt",), msg=""):
    """Validates if a file or files have an acceptable suffix, raising an error if not."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def model_type(p="path/to/model.pt"):
    """
    Determines model type from file path or URL, supporting various export formats.

    Example: path='path/to/model.onnx' -> type=onnx
    """
    # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
    sf = list(export_formats().Suffix)  # export suffixes
    check_suffix(p, sf)  # checks
    types = [s in Path(p).name for s in sf]
    types[8] &= not types[9]  # tflite &= not edgetpu
    return types


def get_file_list(source_root: str) -> [list, dict]:
    """Get file list.

    Args:
        source_root (str): image or video source path

    Return:
        source_file_path_list (list): A list for all source file.
        source_type (dict): Source type: file or url or dir.
    """
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(("http:/", "https:/"))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(
            source_root, IMG_EXTENSIONS, recursive=True, case_sensitive=False
        ):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(urllib.parse.unquote(source_root).split("?")[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f"Downloading source file to {file_save_path}")
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print("Cannot find image file.")

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type
