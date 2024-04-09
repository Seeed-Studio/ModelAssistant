# Copyright (c) Seeed Tech Ltd. All rights reserved.
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

from sscma.registry import LOSSES


@LOSSES.register_module()
class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, torch.tensor(targets.data.unsqueeze(1), dtype=torch.int64), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = targets.cuda()
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss
