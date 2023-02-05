import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def nll_loss(pred, target):
    return F.nll_loss(pred, target, reduction='none')


@LOSSES.register_module()
class NLLLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * nll_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
