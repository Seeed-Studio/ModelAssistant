import os
import os.path as osp
import sys
import shutil
import argparse

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.evaluator import DumpResults

from sscma.utils import lazy_import


def parse_args():
    parser = argparse.ArgumentParser(description="test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="dump predictions to a pickle file for offline evaluation",
    )
    parser.add_argument(
        "--img-size",
        "--img_size",
        "--imgsz",
        nargs="+",
        type=int,
        default=[320, 320],
        help="Image size of height and width",
    )
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify onnx model by onnx-sim"
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument(
        "--image_path", type=str, help="Used to export verification data of tflite"
    )
    parser.add_argument(
        "--format",
        nargs="*",
        default=["onnx"],
        choices=["onnx", "tflite", "vela", "savemodel", "torchscript", "hailo"],
        help="Model format to be exported",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="hailo8l",
        choices=["hailo8", "hailo8l", "hailo15", "hailo15l"],
        help="hailo hardware type",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify whether the exported tflite results are aligned with the tflitemicro results",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
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
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args

def find_and_sample_images(folder_path, limit=10000, sample_size=100):
    import random

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []

    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided image path '{folder_path}' is not a valid directory.")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
                if len(image_files) >= limit:
                    break
        if len(image_files) >= limit:
            break

    found = len(image_files)
    if found < sample_size:
        if found == 0:
            raise ValueError(f"No images found in the directory '{folder_path}'.")
        print(f"Warning: Found only {found} images, which is less than the requested sample size of {sample_size}.")
        sample_size = found

    random_sample = random.sample(image_files, sample_size)

    return random_sample

def generate_input(images_path, img_shape):
    import cv2

    res = []
    datasets = find_and_sample_images(images_path, limit=10000, sample_size=100)
    for ps in datasets[:100]:
        img = cv2.imread(ps)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        img = cv2.resize(img, (img_shape[0], img_shape[1]))
        res.append(img)
        res.append(img.astype(np.float32))
    return np.asarray(res)


def main():
    args = parse_args()
    # verify args
    if (
        True in [fm in ["hailo", "tflite", "vela"] for fm in args.format]
        and args.image_path is None
    ):
        raise ValueError("image_path is required for hailo/tflite/vela format")
    # load config
    cfg = Config.fromfile(args.config, modified_constant=args.cfg_options)
    cfg.launcher = args.launcher
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

    cfg.load_from = args.checkpoint

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
        shutil.rmtree(runner.log_dir, ignore_errors=True)
        runner._log_dir = osp.dirname(args.checkpoint)

    # else:
    #     # build customized runner from the registry
    #     # if 'runner_type' is set in the cfg
    #     runner = RUNNERS.build(cfg)

    runner.call_hook("before_run")
    runner.model.to(device=args.device)
    runner.load_checkpoint(args.checkpoint, map_location=torch.device(args.device))
    model = runner.model.to(device=args.device)

    model_format = args.format
    new_model_format = []
    for fmt in model_format:
        if fmt == "onnx":
            new_model_format.append("onnx")
        elif fmt == "hailo":
            new_model_format.extend(["onnx", "hailo"])
        elif fmt == "savemodel":
            new_model_format.extend(["onnx", "savemodel"])
        elif fmt == "tflite":
            new_model_format.extend(["onnx", "savemodel", "tflite"])
        elif fmt == "vela":
            new_model_format.extend(["onnx", "savemodel", "tflite", "vela"])
        elif fmt == "torchscript":
            new_model_format.append("torchscript")
    new_model_format = list(set(new_model_format))

    calibration_data = None
    if args.image_path:
        if not osp.exists("calibration_image_sample_data_20x128x128x3_float32.npy"):
            input_data = generate_input(args.image_path, args.img_size)
            np.save(
                "calibration_image_sample_data_20x128x128x3_float32.npy", input_data
            )
        std = (
            (
                cfg.model.data_preprocessor.std
                if cfg.model.data_preprocessor.get("std", False)
                else [0, 0, 0]
            )
            if cfg.model.get("data_preprocessor", False)
            else [0, 0, 0]
        )
        mean = (
            (
                cfg.model.data_preprocessor.mean
                if cfg.model.data_preprocessor.get("mean", False)
                else [255, 255, 255]
            )
            if cfg.model.get("data_preprocessor", False)
            else [255, 255, 255]
        )
        calibration_data = [
            [
                "images",
                "calibration_image_sample_data_20x128x128x3_float32.npy",
                [[[std]]],
                [[[mean]]],
            ]
        ]
    # export
    if "torchscript" in new_model_format:
        export_torchscript(model, args)

    if "onnx" in new_model_format:
        onnx_file = export_onnx(model, args)

    if "hailo" in new_model_format:
        export_hailo(onnx_file, args.arch, args.img_size, cfg, args.image_path)

    if "savemodel" in new_model_format:
        export_savemodel(onnx_file, calibration_data)

    if "tflite" in new_model_format:
        tflite_file = export_tflite(
            onnx_file,
            args.img_size,
            args.image_path,
        )

    if "vela" in new_model_format:
        export_vela(tflite_file, args.verify)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(
            (".pkl", ".pickle")
        ), "The dump file must be a pkl file."
        runner.test_evaluator.metrics.append(DumpResults(out_file_path=args.out))


@lazy_import("onnx2tf", install_only=True)
@lazy_import("tf-keras", install_only=True)
@lazy_import("onnx-graphsurgeon", install_only=True)
@lazy_import("sng4onnx", install_only=True)
@lazy_import("onnxsim", install_only=True)
def export_savemodel(onnx_file, calibration_data=None):
    # onnx convert to pb
    from onnx2tf import onnx2tf

    try:
        onnx2tf.convert(
            onnx_file,
            output_folder_path=osp.dirname(onnx_file),
            # batch_size=1,
            custom_input_op_name_np_data_path=calibration_data,
            output_signaturedefs=True,
            verbosity="warn",
        )
        print("The pb model format was exported successfully")
    except Exception as e:
        print("Export of pb model failed, export interrupted")
        raise RuntimeError(e)

    return osp.dirname(onnx_file)


def export_torchscript(model, args):
    from torch.utils.mobile_optimizer import optimize_for_mobile

    f = f"{osp.splitext(args.checkpoint)[0]}_script.pt"
    script_model = torch.jit.trace(
        model, torch.randn(1, 3, *args.img_size).to(args.device)
    )
    script_model = optimize_for_mobile(script_model)
    torch.jit.save(script_model, f)


@lazy_import("onnx")
@lazy_import("onnxsim", install_only=True)
def export_onnx(model, args):
    import onnx

    fake_input = torch.randn(1, 3, *args.img_size).to(args.device)
    f = f"{osp.splitext(args.checkpoint)[0]}.onnx"
    torch.onnx.export(
        model,
        fake_input,
        f,
        verbose=False,
        input_names=["images"],
        opset_version=args.opset,
    )
    onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    if args.simplify:
        try:
            import onnxsim

            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "assert check failed"
            onnx.save(onnx_model, f)
        except Exception as e:
            print(f"Simplify failure: {e}")
            raise RuntimeError(e)

    return f


def export_hailo(onnx_path: str, arch: str, img_shape, cfg, img_path):
    from hailo_sdk_client import ClientRunner
    import onnx
    import cv2
    from hailo_sdk_client.exposed_definitions import CalibrationDataType

    datasets = [
        osp.join(img_path, i) for i in os.listdir(img_path) if i.endswith(".jpg")
    ]
    calib_dataset = []
    for ps in datasets[:300]:
        img = cv2.imread(ps)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        img = cv2.resize(img, (img_shape[0], img_shape[1]))
        calib_dataset.append(img)
    calib_dataset = np.asarray(calib_dataset)

    har_file = f"{osp.dirname(onnx_path)}{osp.sep}{osp.splitext(osp.basename(onnx_path))[0]}.har"
    har_quant_file = f"{osp.dirname(onnx_path)}{osp.sep}{osp.splitext(osp.basename(onnx_path))[0]}_quant.har"
    hef_file = f"{osp.dirname(onnx_path)}{osp.sep}{osp.splitext(osp.basename(onnx_path))[0]}.hef"

    model = onnx.load(onnx_path)
    runner = ClientRunner(hw_arch=arch)
    input_shape = {
        inp.name: [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        for inp in model.graph.input
    }
    runner.translate_onnx_model(
        onnx_path,
        "onnx",
        start_node_names=[i.name for i in model.graph.input],
        end_node_names=[i.name for i in model.graph.output],
        net_input_shapes=input_shape,
    )
    runner.save_har(har_file)
    runner = ClientRunner(har=har_file, hw_arch="hailo8l")

    alls = f"normalization1 = normalization({cfg.model.data_preprocessor.mean}, {cfg.model.data_preprocessor.std})\n"
    runner.load_model_script(alls)
    runner.optimize(calib_dataset, CalibrationDataType.np_array)
    runner.save_har(har_quant_file)
    runner = ClientRunner(har=har_quant_file, hw_arch="hailo8l")
    hef = runner.compile()
    with open(hef_file, "wb") as f:
        f.write(hef)


@lazy_import("tensorflow")
def export_tflite(onnx_path: str, img_shape, img_path):
    import os.path as osp
    import cv2
    from tqdm.std import tqdm
    import tensorflow as tf

    file_stem = osp.splitext(osp.basename(onnx_path))[0]
    tflite_path = osp.join(osp.dirname(onnx_path), f"{file_stem}_int8.tflite")
    # pb convert to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(osp.dirname(onnx_path))

    def representative_dataset():
        datasets = find_and_sample_images(img_path, limit=10000, sample_size=300)
        for ps in tqdm(datasets[:300]):
            img = cv2.imread(ps)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
            img = cv2.resize(img, (img_shape[0], img_shape[1]))
            img = tf.convert_to_tensor([img], dtype=tf.float32)

            yield [img]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset
    converter._experimental_disable_per_channel = False

    tflite_quant_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)
    print("tflite model export successful")

    return tflite_path


@lazy_import("ethos-u-vela", install_only=True)
def export_vela(tflite_path: str, verify=False):
    # tflite convert to vela.tflite
    cmd = f"vela \
    --config {osp.dirname(osp.abspath(__file__))}/vela_config.ini \
    --accelerator-config ethos-u55-64 \
    --verbose-performance \
    --system-config My_Sys_Cfg \
    --memory-mode My_Mem_Mode_Parent \
    --output-dir {osp.dirname(tflite_path)}/ \
    {tflite_path}"
    state = os.system(cmd)
    if not state:
        print("Export of vela model succeeded")
    else:
        raise RuntimeError("Export of vela model failed")

    if verify:
        verify_tflite(tflite_path)


def verify_tflite(tflite_path):
    if sys.version_info.major > 2 and sys.version_info.minor > 9:
        import math
        import numpy as np
        import tensorflow as tf
        import tflite_micro as tflm

        sys.setrecursionlimit(1000000)

        tfl_interpreter = tf.lite.Interpreter(
            model_path=tflite_path, experimental_preserve_all_tensors=True
        )
        tfl_interpreter.allocate_tensors()
        input_image = np.random.randint(
            0, 255, tfl_interpreter.get_input_details()[0]["shape_signature"]
        )

        tflm_interpreter = tflm.runtime.Interpreter.from_file(
            tflite_path,
            intrepreter_config=tflm.runtime.InterpreterConfig.kPreserveAllTensors,
        )

        tfl_interpreter.set_tensor(
            tfl_interpreter.get_input_details()[0]["index"], input_image
        )
        tfl_interpreter.invoke()

        tflm_interpreter.set_input(input_image, 0)
        tflm_interpreter.invoke()
        for i, details in enumerate(tfl_interpreter.get_output_details()):
            tflm_tensor = tflm_interpreter.get_output(i)
            tfl_tensor = tfl_interpreter.get_tensor(details["index"])
            is_match = np.allclose(tfl_tensor, tflm_tensor, atol=1, equal_nan=True)
            Accuracy = np.sum(tfl_tensor == tflm_tensor) / math.prod(tfl_tensor.shape)
            string = f'name:{details["name"]} shape:{details["shape"]} Accuracy:{Accuracy} match:{is_match}'
            print(string)

    else:
        print(
            "Using tflite micro requires your Python version to be 3.10 or above.",
            f" Your Python version {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            " cannot be verified and has been skipped.",
        )


if __name__ == "__main__":
    main()
