# Copyright (c) Seeed Technology Co.,Ltd.
"""Deployment inference wrapper for Swift-YOLO (YOLOv5-style) models.

The exported graph (see DetHead._forward/process in yolov5_head.py) emits a
single decoded tensor of shape [batch, num_priors, 4 + 1 + num_classes]:

- ``[..., 0:2]``: box center x/y in pixels of the network input
- ``[..., 2:4]``: box width/height in pixels of the network input
- ``[..., 4]``: objectness score, scaled by 100
- ``[..., 5:]``: class scores, scaled by 100

(the x100 scaling is the SSCMA firmware convention shared with the 2.0.0
branch). This wrapper converts that tensor into standard DetDataSample
predictions: xyxy boxes, obj*cls scores, top-k filtering, NMS and rescaling
to the original image.
"""

import copy
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from mmengine import MODELS
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from sscma.structures import DetDataSample, OptSampleList
from sscma.utils.misc import filter_scores_and_topk, samplelist_boxtype2tensor
from sscma.utils.typing_utils import OptConfigType, OptMultiConfig

from ..backend import BaseInfer

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor]


class YOLOInfer(BaseModel):
    """YOLOInfer class for Swift-YOLO (YOLOv5-style) serial inference.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`. it usually includes,
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
        """The unified entry for a forward process. Only "predict" mode is
        supported: forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`."""
        if mode == "predict":
            return self._predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' "YOLOInfer Only supports predict mode")

    def _predict(self, inputs: torch.Tensor, batch_data_samples: OptSampleList):
        """Predict results from a batch of inputs and data samples with
        post-processing."""
        data_tmp = self.func.infer(inputs)
        # the backends run split inference: a list with one output-list per
        # image; flatten back to [batch, num_priors, 4 + 1 + num_classes]
        if isinstance(data_tmp, (list, tuple)):
            outs = []
            for item in data_tmp:
                arr = item[0] if isinstance(item, (list, tuple)) else item
                outs.append(torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr)
            data_tmp = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
        elif isinstance(data_tmp, np.ndarray):
            data_tmp = torch.from_numpy(data_tmp)
        # [batch, num_priors, 4 + 1 + num_classes]
        assert data_tmp.dim() == 3, f"unexpected model output shape {tuple(data_tmp.shape)}"

        cfg = self.config.model.test_cfg
        score_thr = cfg.get("score_thr", -1)
        nms_pre = cfg.get("nms_pre", 100000)
        multi_label = cfg.get("multi_label", True) and self.pred_head.num_classes > 1

        for preds, data_sample in zip(data_tmp, batch_data_samples):
            xywh, obj, cls_scores = preds[:, :4], preds[:, 4] / 100.0, preds[:, 5:] / 100.0

            if score_thr > 0:
                conf_inds = obj > score_thr
                xywh, obj, cls_scores = xywh[conf_inds], obj[conf_inds], cls_scores[conf_inds]

            # conf = obj_conf * cls_conf
            scores = cls_scores * obj[:, None]

            # xywh center -> xyxy corners
            bboxes = torch.empty_like(xywh)
            bboxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
            bboxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
            bboxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
            bboxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

            if scores.shape[0] == 0:
                data_sample.pred_instances = InstanceData(
                    bboxes=bboxes, scores=scores[:, 0] if scores.numel() else scores.new_zeros(0),
                    labels=scores.new_zeros(0, dtype=torch.int64),
                )
                continue

            if not multi_label:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores, score_thr, nms_pre, results=dict(labels=labels[:, 0])
                )
                labels = results["labels"]
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(scores, score_thr, nms_pre)

            results = InstanceData(scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            img_meta = data_sample.metainfo
            ori_shape = img_meta["ori_shape"]
            pad_param = img_meta.get("pad_param", None)
            if pad_param is not None:
                results.bboxes -= results.bboxes.new_tensor([pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
            scale_factor = img_meta.get("scale_factor", None)
            if scale_factor is not None:
                results.bboxes /= results.bboxes.new_tensor(scale_factor).repeat((1, 2))

            results = self.pred_head._bbox_post_process(
                results=results, cfg=cfg, rescale=False, with_nms=True, img_meta=img_meta
            )
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            data_sample.pred_instances = results

        samplelist_boxtype2tensor(batch_data_samples)
        return batch_data_samples

    def set_infer(self, func: BaseInfer, Config: OptConfigType = None):
        self.func = func
        self.func.load_weights()
        if Config is not None:
            self.config = copy.deepcopy(Config)
            self.pred_head = MODELS.build(self.config.model.bbox_head)
