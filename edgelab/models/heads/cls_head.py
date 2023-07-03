from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from edgelab.registry import MODELS

from mmcls.models.heads.cls_head import ClsHead as MMClsHead


@MODELS.register_module()
class Audio_head(nn.Module):

    def __init__(self, in_channels, n_classes, drop=0.5):
        super(Audio_head, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, in_channels)
        self.fc1 = nn.Linear(in_channels, n_classes)
        self.dp = nn.Dropout(drop)

    def forward(self, x):
        return self.fc1(self.dp(self.fc(self.avg(x).flatten(1))))


@MODELS.register_module()
class ClsHead(MMClsHead):
    def __init__(self,
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk: Union[int, Tuple[int]] = (1, ),
                 cal_acc: bool = False,
                 softmax: bool = False,
                 init_cfg: Optional[dict] = None):
        super(ClsHead, self).__init__(loss, topk, cal_acc, init_cfg)
        
        self.softmax = softmax
        
    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
   
        pre_logits = self.pre_logits(feats)
        # The ClsHead doesn't have the final classification head,
        # just return the unpacked inputs.
        if(self.softmax):
            pre_logits = F.softmax(pre_logits, dim=1)
        return pre_logits
        
        
    