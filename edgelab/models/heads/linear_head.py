from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from edgelab.registry import MODELS

from mmcls.models.heads.linear_head import LinearClsHead as MMLinearClsHead

@MODELS.register_module()
class LinearClsHead(MMLinearClsHead):
    def __init__(self,
                 softmax: bool = True,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)
        
        self.softmax = softmax
        
    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        
        if(self.softmax):
            cls_score = F.softmax(cls_score, dim=1)
        
        return cls_score