import argparse
import os
import tempfile

import torch
from loguru import logger

# TODO: Move to config file
# import edgelab.datasets
# import edgelab.engine
# import edgelab.evaluation
# import edgelab.models
# import edgelab.visualization


def parse_args():
    from mmengine.config import DictAction

    parser = argparse.ArgumentParser(description="Train EdgeLab models")

    # common configs
    parser.add_argument("config", type=str, help="the model config file path")
    parser.add_argument(
        "--work_dir",
        "--work-dir",
        type=str,
        default=None,
        help="the directory to save logs and models",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision during training (https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)",
    )
    parser.add_argument(
        "--auto_scale_lr",
        "--auto-scale-lr",
        action="store_true",
        default=False,
        help="enable automatic-scale-LR during training",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        const="auto",
        help="resume training from the checkpoint of the last epoch (or a specified checkpoint path)",
    )
    parser.add_argument(
        "--no_validate",
        "--no-validate",
        action="store_true",
        default=False,
        help="disable checkpoint evaluation during training",
    )
    parser.add_argument(
        "--no_persistent_workers",
        "--no-persistent-workers",
        action="store_true",
        default=False,
        help="disable persistent workers for dataloaders",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="the device used for convert & export",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        default="none",
        choices=["none", "pytorch", "slurm", "mpi"],
        help="the job launcher for MMEngine",
    )
    parser.add_argument(
        "--cfg_options",
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair in 'xxx=yyy' format will be merged into config file",
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=0,
        help="set local-rank for PyTorch",
    )
    parser.add_argument(
        "--dynamo_cache_size",
        "--dynamo-cache-size",
        type=int,
        default=None,
        help="set dynamo-cache-size limit for PyTorch",
    )

    # extension
    parser.add_argument(
        "--input_shape",
        "--input-shape",
        type=int,
        nargs="+",
        default=None,
        help="Extension: input data shape for model parameters estimation, e.g. 1 3 224 224",
    )

    return parser.parse_args()


def verify_args(args):
    assert os.path.splitext(args.config)[-1] == ".py", "The config file name should be ended with a '.py' extension"
    assert os.path.exists(args.config), "The config file does not exist"
    assert args.local_rank >= 0, "The local-rank should be larger than or equal to 0"
    if args.dynamo_cache_size is not None:
        assert args.dynamo_cache_size > 0, "The local-rank should be larger than or equal to 0"

    return args


def build_config(args):
    from mmengine.config import Config

    from tools.utils.config import load_config

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.dynamo_cache_size is not None:
        torch._dynamo.config.cache_size_limit = args.dynamo_cache_size

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_data = load_config(args.config, folder=tmp_dir, cfg_options=args.cfg_options)
        cfg = Config.fromfile(cfg_data)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.launcher = args.launcher

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        args.work_dir = cfg.work_dir = os.path.join("work_dirs", os.path.splitext(os.path.basename(args.config))[0])

    if args.device.startswith("cuda"):
        args.device = args.device if torch.cuda.is_available() else "cpu"

    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get("type", "OptimWrapper")
        assert optim_wrapper in [
            "OptimWrapper",
            "AmpOptimWrapper",
        ], f"automatic-mixed-precision is not supported by {optim_wrapper}"
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.setdefault("loss_scale", "dynamic")

    if args.resume == "auto":
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    for dataloader in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        if cfg.get(dataloader, None) is None:
            logger.warning("Skipped setting parms for empty dataloader: {}", dataloader)
            continue
        for k, i_v in {
            "persistent_workers": args.no_persistent_workers,
        }.items():
            if i_v is None:
                continue
            if cfg.get(cfg[dataloader][k], None) is None:
                logger.warning(
                    "Skipped setting dataloader parms for missing attribute: {}.{}",
                    dataloader,
                    k,
                )
                continue
            cfg[dataloader][k] = not i_v

    if args.input_shape is None:
        try:
            if "shape" in cfg:
                args.input_shape = cfg.shape
            elif "width" in cfg and "height" in cfg:
                args.input_shape = [
                    1,
                    3,
                    cfg.width,
                    cfg.height,
                ]
        except Exception as exc:
            raise ValueError("Please specify the input shape") from exc
        logger.warning(
            "Using automatically generated input shape (from config '{}'): {}",
            os.path.basename(args.config),
            args.input_shape,
        )

    return args, cfg


@logger.catch(onerror=lambda _: os._exit(1))
def main():
    args = parse_args()
    args = verify_args(args)
    args, cfg = build_config(args)

    if "runner_type" not in cfg:
        from mmengine.runner import Runner

        runner = Runner.from_cfg(cfg)
    else:
        from mmengine.registry import RUNNERS

        runner = RUNNERS.build(cfg)

    dummy_inputs = torch.randn(*args.input_shape, device=args.device)
    model = runner.model.to(device=args.device)
    model.eval()
    from mmengine.analysis import get_model_complexity_info

    analysis_results = get_model_complexity_info(model=model, input_shape=args.input_shape, inputs=(dummy_inputs,))

    logger.info("Model Flops:{}", analysis_results["flops_str"])
    logger.info("Model Parameters:{}", analysis_results["params_str"])

    runner.train()


if __name__ == "__main__":
    main()
