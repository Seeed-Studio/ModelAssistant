# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from typing import Optional, Tuple, Union

import torch

from sscma.registry import MODELS

from .cls_head import ClsHead


@MODELS.register_module()
class AxesClsHead(ClsHead):
    def __init__(
        self,
        loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk: Union[int, Tuple[int]] = (1,),
        cal_acc: bool = False,
        init_cfg: Optional[dict] = None,
    ):
        super(AxesClsHead, self).__init__(loss, topk, cal_acc, init_cfg=init_cfg)

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""

        pre_logits = self.pre_logits(feats)
        # The ClsHead doesn't have the final classification head,
        # just return the unpacked inputs.
        return pre_logits
