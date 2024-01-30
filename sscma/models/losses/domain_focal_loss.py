from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from sscma.registry import LOSSES


@LOSSES.register_module()
class DomainFocalLoss(nn.Module):
    def __init__(
        self,
        class_num: int,
        alpha: Optional[Union[float, Variable]] = None,
        gamma: int = 2,
        size_average: bool = True,
        sigmoid: bool = False,
        reduce: bool = True,
    ) -> None:
        super(DomainFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        N = inputs.size(0)

        C = inputs.size(1)
        if self.sigmoid:
            P = F.sigmoid(inputs)
            # F.softmax(inputs)
            if targets == 0:
                probs = 1 - P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        else:
            P = F.softmax(inputs, dim=1)

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.0)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


@LOSSES.register_module()
class TargetLoss:
    def __init__(self) -> None:
        self.fl = DomainFocalLoss(class_num=2)

    def __call__(self, feature: torch.Tensor) -> torch.Tensor:
        out_8 = feature[0]
        out_16 = feature[1]
        out_32 = feature[2]

        out_d_t_8 = out_8.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_t_16 = out_16.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_t_32 = out_32.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_t = torch.cat((out_d_t_8, out_d_t_16, out_d_t_32), 0)

        # domain label
        domain_t = Variable(torch.ones(out_d_t.size(0)).long().cuda())
        dloss_t = 0.5 * self.fl(out_d_t, domain_t)
        return dloss_t


@LOSSES.register_module()
class DomainLoss:
    def __init__(self):
        self.fl = DomainFocalLoss(class_num=2)

    def __call__(self, feature: Tuple[torch.Tensor]) -> torch.Tensor:
        out_8 = feature[0]
        out_16 = feature[1]
        out_32 = feature[2]

        out_d_s_8 = out_8.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_s_16 = out_16.permute(0, 2, 3, 1).reshape(-1, 2)
        out_d_s_32 = out_32.permute(0, 2, 3, 1).reshape(-1, 2)

        out_d_s = torch.cat((out_d_s_8, out_d_s_16, out_d_s_32), 0)

        # domain label
        domain_s = Variable(torch.zeros(out_d_s.size(0)).long().cuda())
        # global alignment loss
        dloss_s = 0.5 * self.fl(out_d_s, domain_s)
        return dloss_s
