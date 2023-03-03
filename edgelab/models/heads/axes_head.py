import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import LinearClsHead
from mmcls.models.utils import is_tracing


@HEADS.register_module()
class AxesClsHead(LinearClsHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(AxesClsHead, self).__init__(num_classes=num_classes, in_channels=in_channels, init_cfg=init_cfg, *args, **kwargs)

        self.fp16_enabled = False
        self.on_trace = False
        
    
    def forward_dummy(self, x, softmax=True):
        
        x = self.pre_logits(x)
        
        cls_score = self.fc(x)
        
        if softmax:
            pred = F.softmax(cls_score)
        else:
            pred = cls_score
            
        
        return pred
            

