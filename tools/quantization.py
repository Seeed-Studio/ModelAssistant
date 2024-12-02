import argparse
import os
import os.path as osp
import time
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
import torch.nn as nn

current_path = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.dirname(current_path))


from tinynn.graph.quantization.quantizer import QATQuantizer, PostQuantizer
from tinynn.util.train_util import AverageMeter
from tinynn.graph.tracer import model_tracer
from tinynn.graph.quantization.algorithm.cross_layer_equalization import (
    cross_layer_equalize,
)
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize
from tinynn.converter import TFLiteConverter
from tinynn.util.quantization_analysis_util import (
    graph_error_analysis,
    layer_error_analysis,
    get_weight_dis,
)
from tinynn.prune.identity_pruner import IdentityChannelPruner


from mmengine.config import Config, DictAction
from mmengine.evaluator import DumpResults
from mmengine.fileio.backends import backends
from mmengine.runner import Runner
from mmengine.device import get_device
from mmengine.model import BaseModel
from sscma.utils.typing_utils import OptConfigType, OptMultiConfig
from sscma.structures import DetDataSample, OptSampleList
from sscma.utils.misc import samplelist_boxtype2tensor

ForwardResults = Union[
    Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor
]


def parse_args():
    parser = argparse.ArgumentParser(description="quantizer a model")
    parser.add_argument("config", help="quantization config file path")
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


class QuantModel(BaseModel):
    """RTMDetInfer class for rtmdet serial inference.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
    """

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        tinynn_model: torch.nn.Module = None,
        bbox_head: torch.nn.Module = None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self._model = tinynn_model
        self.bbox_head = bbox_head

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: OptSampleList = None,
        mode: str = "predict",
    ) -> ForwardResults:
        """The unified entry for a forward process in both training and test.
        The method should accept three modes: "tensor", "predict" and "loss":

        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.
        """
        if mode == "predict":
            data = self._model(inputs)
            batch_img_metas = [data_samples.metainfo for data_samples in data_samples]
            results = self.bbox_head.predict_by_feat(
                *data, batch_img_metas=batch_img_metas
            )
            # data_samples.pred_instances = result
            for result, data_sample in zip(results, data_samples):
                data_sample.pred_instances = result

            samplelist_boxtype2tensor(data_samples)
            return data_samples
        elif mode == "loss":
            return self._loss(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". ' "QuantModel Only supports predict mode"
            )

    def _loss(self, inputs: torch.Tensor, batch_data_samples: OptSampleList):
        data = self._model(inputs)
        # Fast version
        loss_inputs = data + (
            batch_data_samples["bboxes_labels"],
            batch_data_samples["img_metas"],
        )
        losses = self.bbox_head.loss_by_feat(*loss_inputs)
        return losses


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
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

    cfg.load_from = args.model

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    runner.test()
    model = runner.model

    # pruner = IdentityChannelPruner(
    #     model, torch.ones(1, 3, 640, 640), config={"multiple": 8}
    # )
    # st_flops = pruner.calc_flops()
    # pruner.prune()  # Get the pruned model
    #
    # print("Validation accuracy of the pruned model")
    # runner.model = model
    # # runner.test()
    # ed_flops = pruner.calc_flops()
    # print(
    #     f"Pruning over, reduced FLOPS {100 * (st_flops - ed_flops) / st_flops:.2f}%  ({st_flops} -> {ed_flops})"
    # )

    with model_tracer():
        model_copy = copy.deepcopy(model).eval()
        model_copy = model_copy.to(get_device())
        # model_copy = model_copy.to("cpu")
        dummy_input = torch.randn(1, 3, 640, 640)
        # model_copy = cross_layer_equalize(model_copy, dummy_input, get_device())

        # More information for QATQuantizer initialization, see `examples/quantization/qat.py`.
        # We set 'override_qconfig_func' when initializing QATQuantizer to use fake-quantize to do post quantization.
        # quantizer = PostQuantizer(
        #     model_copy,
        #     dummy_input,
        #     work_dir="out",
        #     config={
        #         "asymmetric": True,
        #         "backend": "qnnpack",
        #         "disable_requantization_for_cat": True,
        #         "per_tensor": True,
        #         "override_qconfig_func": set_ptq_fake_quantize,
        #     },
        # )
        # per tensor quantization with out cle : 0.254

        quantizer = QATQuantizer(
            model_copy,
            dummy_input,
            work_dir="out",
            config={
                "asymmetric": True,
                "force_overwrite": True,
                "per_tensor": False,
                "override_qconfig_func": set_ptq_fake_quantize,
            },
        )
        ptq_model = quantizer.quantize()

        quantizer.optimize_conv_bn_fusion(ptq_model)

    # with torch.no_grad():
    #     ptq_model.eval()
    #     ptq_model.cpu()
    #
    #     # Post quantization calibration
    #     ptq_model.apply(torch.quantization.disable_fake_quant)
    #     ptq_model.apply(torch.quantization.enable_observer)
    #
    #     calibrate(ptq_model, runner, max_iteration=10, device="cpu")
    #
    #     # Disable observer and enable fake quantization to validate model with quantization error
    #     ptq_model.apply(torch.quantization.disable_observer)
    #     ptq_model.apply(torch.quantization.enable_fake_quant)
    #
    #     datas = next(iter(runner.train_dataloader))
    #     datas = runner.model.data_preprocessor(datas, False)
    #     dummy_input_real = datas["inputs"][1].squeeze(0)
    #     graph_error_analysis(ptq_model, dummy_input_real, metric="cosine")
    #
    #     layer_error_analysis(ptq_model, dummy_input_real, metric="cosine")
    #
    # exit(0)

    # with torch.no_grad():
    #     ptq_model.eval()
    #     ptq_model.cpu()
    #     # -----------------------------------------------
    #     for n, m in model.named_modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             f = m.weight / (m.running_var**0.5)
    #             print(f"Layer: {n}, Min: {f.min()}, Max: {f.max()}")

    q_model = QuantModel(
        data_preprocessor=runner.model.data_preprocessor,
        bbox_head=runner.model.bbox_head,
        tinynn_model=ptq_model,
    )
    runner.model = q_model
    runner.test()

    runner.train()

    with torch.no_grad():
        ptq_model.eval()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        ptq_model.cpu()
        ptq_model = quantizer.convert(ptq_model, backend="qnnpack")

        tf_converter = TFLiteConverter(
            ptq_model,
            torch.randn(1, 3, 640, 640),
            tflite_path="out/qat_model.tflite",
            fuse_quant_dequant=True,
            quantize_target_type="int8",
        )
        tf_converter.convert()


if __name__ == "__main__":
    main()
