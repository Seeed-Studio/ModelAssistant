import torch
import torch.nn as nn
import torch.nn.functional as F
from edgelab.registry import BACKBONES


class ResBlock1dTF(nn.Module):

    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        self.block_t = nn.Sequential(
            # nn.ReflectionPad1d(dilation * (kernel_size//2)),
            nn.Conv1d(dim,
                      dim,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=dilation * (kernel_size // 2),
                      bias=False,
                      dilation=dilation,
                      groups=dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True))
        self.block_f = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, bias=False),
                                     nn.BatchNorm1d(dim),
                                     nn.LeakyReLU(0.2, True))
        self.shortcut = nn.Conv1d(dim, dim, 1, 1)

    def forward(self, x):
        return self.shortcut(x) + self.block_f(x) + self.block_t(x)


class AADownsample(nn.Module):

    def __init__(self, filt_size=3, stride=2, channels=None):
        super(AADownsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        ha = torch.arange(1, filt_size // 2 + 1 + 1, 1)
        a = torch.cat((ha, ha.flip(dims=[
            -1,
        ])[1:])).float()
        a = a / a.sum()
        filt = a[None, :]
        self.register_buffer('filt', filt[None, :, :].repeat(
            (self.channels, 1, 1)))

    def forward(self, x):
        # x_pad = F.pad(x, (self.filt_size//2, self.filt_size//2))
        y = F.conv1d(x,
                     self.filt,
                     stride=self.stride,
                     padding=self.filt_size // 2,
                     groups=x.shape[1])
        return y


class Down(nn.Module):

    def __init__(self, channels, d=2, k=3):
        super().__init__()
        kk = d + 1
        self.down = nn.Sequential(
            # nn.ReflectionPad1d(kk // 2),
            nn.Conv1d(channels,
                      channels * 2,
                      kernel_size=kk,
                      stride=1,
                      padding=kk // 2,
                      bias=False),
            nn.BatchNorm1d(channels * 2),
            nn.LeakyReLU(0.2, True),
            AADownsample(channels=channels * 2, stride=d, filt_size=k))

    def forward(self, x):
        x = self.down(x)
        return x


@BACKBONES.register_module()
class SoundNetRaw(nn.Module):

    def __init__(self,
                 nf=2,
                 clip_length=None,
                 factors=[4, 4, 4],
                 out_channel=32):
        super().__init__()
        base_ = 4
        model = [
            # nn.ReflectionPad1d(3),
            nn.Conv1d(1, nf, kernel_size=11, stride=6, padding=5, bias=False),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True),
        ]
        self.start = nn.Sequential(*model)
        model = []
        for i, f in enumerate(factors):
            model += [Down(channels=nf, d=f, k=f * 2 + 1)]
            nf *= 2
            if i % 2 == 0:
                model += [ResBlock1dTF(dim=nf, dilation=1, kernel_size=7)]
        self.down = nn.Sequential(*model)

        factors = [2, 2]
        model = []
        for _, f in enumerate(factors):
            for i in range(1):
                for j in range(3):
                    model += [
                        ResBlock1dTF(dim=nf, dilation=3**j, kernel_size=7)
                    ]
            model += [Down(channels=nf, d=f, k=f * 2 + 1)]
            nf *= 2
        self.down2 = nn.Sequential(*model)
        self.project = nn.Conv1d(nf, out_channel, 1)
        self.clip_length = clip_length
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            with torch.no_grad():
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.cuda()
        x = self.start(x)
        x = self.down(x)
        x = self.down2(x)
        feature = self.project(x)
        return feature


if __name__ == '__main__':
    pass
