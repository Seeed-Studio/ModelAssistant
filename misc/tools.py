import torch
from torch import nn


class CustomMSELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        # 计算损失
        loss = (input - target) * (input - target)

        # 根据 reduction 参数进行处理
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def psnr(x, label):
    # 接受x为张量输入
    loss = CustomMSELoss(reduction='mean')
    psnr = 20 * torch.log10(1 / torch.sqrt(loss(x, label)))
    return psnr
