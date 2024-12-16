from typing import Union, List, Dict, Tuple
import torch

from mmengine.registry import MODELS
from mmengine.model import BaseModel
from sscma.utils.typing_utils import OptConfigType, OptMultiConfig
from sscma.structures import DetDataSample, OptSampleList
from sscma.utils.misc import samplelist_boxtype2tensor

ForwardResults = Union[
    Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor
]


class RtmdetQuantModel(BaseModel):
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
        self.bbox_head = MODELS.build(bbox_head)
        # self.bbox_head = bbox_head

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
    
    def set_model(self,model):
        self._model=model