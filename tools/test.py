import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner
from mmengine.registry import MODELS
from sscma.deploy.backend import (
    TorchScriptInfer,
    OnnxInfer,
    SavedModelInfer,
    TFliteInfer,
)
from sscma.deploy.utils import model_type


def parse_args():
    parser = argparse.ArgumentParser(description="test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("model", help="checkpoint file")
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
        "--show", action="store_true", help="Whether to visualize inference results"
    )
    parser.add_argument(
        "--show_dir",
        "--show-dir" "-o",
        type=str,
        help="Path to save visualization results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="dump predictions to a pickle file for offline evaluation",
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
    cfg = Config.fromfile(args.config, modified_constant=args.cfg_options)
    cfg.launcher = args.launcher
    cfg.custom_hooks = []

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

    if args.show:
        cfg.default_hooks.visualization.show = True

    if args.show_dir:
        cfg.default_hooks.visualization.test_out_dir = args.show_dir

    cfg.load_from = args.model
    # cfg.model = cfg.deploy
    # build model
    model = MODELS.build(cfg.deploy)

    # select backend
    backend = model_type(args.model)
    if backend[2]:  # torchscript
        infer_torchscript_model = TorchScriptInfer(args.model)
        model.set_infer(infer_torchscript_model, cfg)
    elif backend[3]:  # onnx
        infer_onnx_model = OnnxInfer(args.model)
        model.set_infer(infer_onnx_model, cfg)
    elif backend[9]:  # TFlite
        infer_tflite_model = TFliteInfer(args.model)
        model.set_infer(infer_tflite_model, cfg)
    elif backend[7]:  # saved_model
        infer_saved_model = SavedModelInfer(args.model)
        model.set_infer(infer_saved_model, cfg)
    elif backend[0] or backend[1]:
        model = MODELS.build(cfg.model)

    runner = Runner.from_cfg(cfg)
    if not (backend[0] or backend[1]):
        runner.load_or_resume = lambda *args: None
        runner.model = model

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(
            (".pkl", ".pickle")
        ), "The dump file must be a pkl file."
        runner.test_evaluator.metrics.append(DumpResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == "__main__":
    main()
