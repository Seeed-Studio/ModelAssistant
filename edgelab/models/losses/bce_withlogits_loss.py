import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def bcewithlogits_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


@MODELS.register_module()
class BCEWithLogitsLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = reduction_override if reduction_override else self.reduction

        loss = bcewithlogits_loss(pred,
                                  target,
                                  weight,
                                  reduction=reduction,
                                  avg_factor=avg_factor)
        return loss
