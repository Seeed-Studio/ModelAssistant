from typing import Union, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from edgelab.registry import LOSSES
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def bcewithlogits_loss(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)


@LOSSES.register_module()
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def __init__(self,
                 weight: Union[Tuple[int or float, ...], Tensor, None] = None,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 pos_weight: Tensor or int or None = None) -> None:
        if isinstance(weight, (int, float)):
            weight = Tensor([weight])

        if isinstance(weight, (list, tuple)):
            weight = Tensor(weight)

        if isinstance(pos_weight, (int, float)):
            pos_weight = Tensor([pos_weight])

        if isinstance(pos_weight, (list, tuple)):
            pos_weight = Tensor(pos_weight)

        super().__init__(weight, size_average, reduce, reduction, pos_weight)
