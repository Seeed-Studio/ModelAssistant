import torch
from torch import nn
import numpy as np


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
    loss = CustomMSELoss(reduction='sum')
    psnr = 20 * torch.log10(1 / torch.sqrt(loss(x, label)))
    # psnr = loss(x, label)
    return psnr


def col_psnr(x, x_label, c, c_label):
    x = torch.tensor(x)
    x_label = torch.tensor(x_label)
    c = torch.tensor(c)
    c_label = torch.tensor(c_label)
    loss1 = psnr(x, x_label)
    loss2 = psnr(c, c_label)
    return loss1, loss2


def rand_genarate(size):
    seed = torch.randn(size)
    seed = seed.numpy()
    np.save('seed.npy', seed)


# rand_genarate((1, 64))
