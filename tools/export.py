import os
import os.path as osp
import sys
import argparse

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.evaluator import DumpResults


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
        choices=[
            "onnx",
            "tflite",
            "vela",
            "savemodel",
            "torchscript",
        ],
        help="Model format to be exported",
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


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
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
        elif fmt == "savemodel":
            new_model_format.extend(["onnx", "savemodel"])
        elif fmt == "tflite":
            new_model_format.extend(["onnx", "savemodel", "tflite"])
        elif fmt == "vela":
            new_model_format.extend(["onnx", "savemodel", "tflite", "vela"])
        elif fmt == "torchscript":
            new_model_format.append("torchscript")
    new_model_format = list(set(new_model_format))

    # export
    if "torchscript" in new_model_format:
        export_torchscript(model, args)

    if "onnx" in new_model_format:
        onnx_file = export_onnx(model, args)

    if "savemodel" in new_model_format:
        export_savemodel(onnx_file)

    if "tflite" in new_model_format:
        tflite_file = export_tflite(onnx_file, args.image_path, args.img_size)

    if "vela" in new_model_format:
        export_vela(tflite_file, args.verify)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(
            (".pkl", ".pickle")
        ), "The dump file must be a pkl file."
        runner.test_evaluator.metrics.append(DumpResults(out_file_path=args.out))


def export_savemodel(onnx_file):
    tflite_path = f"{osp.splitext(onnx_file)[0]}_int8.tflite"

    # onnx convert to pb
    cmd = f"onnx2tf -i {onnx_file}  -v warn  -osd -o {osp.dirname(onnx_file)}"
    state = os.system(cmd)
    if not state:
        print("The pb model format was exported successfully")
    else:
        print("Export of pb model failed, export interrupted")
        return

    return osp.dirname(onnx_file)


def export_torchscript(model, args):
    from torch.utils.mobile_optimizer import optimize_for_mobile

    f = f"{osp.splitext(args.checkpoint)[0]}_script.pt"
    script_model = torch.jit.script(model)
    script_model = optimize_for_mobile(script_model)
    torch.jit.save(script_model, f)


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

    return f


def export_tflite(onnx_path: str, img_path, img_shape):
    import os.path as osp
    import cv2
    from tqdm.std import tqdm
    import tensorflow as tf

    tflite_path = f"{osp.splitext(onnx_path)[0]}_int8.tflite"
    # pb convert to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(osp.dirname(onnx_path))

    def representative_dataset():
        datasets = [
            osp.join(img_path, i) for i in os.listdir(img_path) if i.endswith(".jpg")
        ]
        for ps in tqdm(datasets[:100]):
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

    # converter._experimental_disable_fuse_mul_and_fc
    # converter.experimental_new_dynamic_range_quantizer=True
    # converter.experimental_use_stablehlo_quantizer=True

    tflite_quant_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)
    print("tflite model export successful")

    return tflite_path


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
        print("Exporting vela model failed")

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
