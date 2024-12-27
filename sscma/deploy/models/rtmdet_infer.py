import copy
import math
import warnings
from typing import Dict, List, Tuple, Union

import torch

from mmengine import MODELS
from mmengine.model import BaseModel
from sscma.structures import DetDataSample, OptSampleList
from sscma.utils.typing_utils import OptConfigType, OptMultiConfig
from sscma.utils.misc import samplelist_boxtype2tensor
from sscma.models.heads import RTMDetHead
from ..backend import BaseInfer

ForwardResults = Union[
    Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor
]


class RTMDetInfer(BaseModel):
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
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.pred_head = None
        self.func = None
        self.config = None

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

            - ``mode="predict"``, return a list of :obj:`DetDataSample`.
        """
        if mode == "predict":
            return self._predict(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". ' "RTMDetInfer Only supports predict mode"
            )

    def _predict(
        self,
        inputs: torch.Tensor,
        batch_data_samples: OptSampleList,
    ):
        """Predict results from a batch of inputs and data samples with post-processing.

        Args:
            inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
        Returns:
            list[:obj:`DetDataSample`]: The predicted data samples.
        """

        data_tmp = self.func.infer(inputs)
        _, _, H, W = inputs.shape
        featmap_size = [(H // s, W // s) for s in self.scale]
        data = []
        if all([d.shape[1] == 4 for d in data_tmp[0]]):
            warnings.warn(
                "Please note that the inference found that the output result channels are all 4, which may lead to wrong results",
                Warning,
            )

        for dt in data_tmp:
            tmp = [None for _ in range(6)]
            for d in dt:
                fs = int(math.sqrt(d.shape[1]))
                ts = (fs, fs)
                if ts in featmap_size:
                    if d.shape[2] == 4:
                        tmp[3 + featmap_size.index(ts)] = d
                    else:
                        tmp[featmap_size.index(ts)] = d

            data.append(tmp)


        for result, data_sample in zip(data, batch_data_samples):
            # check item in result is tensor or numpy

            if isinstance(result[0][0], torch.Tensor):
                cls_scores = result[0]
                bbox_preds = result[1]
            else:
                cls_scores = [torch.from_numpy(item) for item in result[:3]]
                bbox_preds = [torch.from_numpy(item) for item in result[3:]]

            result = self.pred_head.predict_by_feat(
                cls_scores,
                bbox_preds,
                objectnesses=None,
                batch_img_metas=[data_sample.metainfo],
                cfg=self.config.model.test_cfg,
            )
            data_sample.pred_instances = result[0]
        samplelist_boxtype2tensor(batch_data_samples)
        return batch_data_samples

    def set_infer(self, func: BaseInfer, Config: OptConfigType = None):
        self.func = func
        self.func.load_weights()
        if Config is not None:
            self.config = copy.deepcopy(Config)
            self.pred_head: RTMDetHead = MODELS.build(self.config.model.bbox_head)
        self.scale = self.pred_head.featmap_strides
