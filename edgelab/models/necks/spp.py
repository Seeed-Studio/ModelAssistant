import torch
import torch.nn as nn

from mmdet.models.builder import NECKS

from ..base.general import CBR


@NECKS.register_module()
class SPP(nn.Module):

    def __init__(self,
                 input_channels,
                 output_channels,
                 layers=[1, 2, 3]) -> None:
        super(SPP, self).__init__()
        self.layers = layers

        self.conv = CBR(input_channels, output_channels, 1, 1, padding=0)
        for idx, value in enumerate(layers):
            layer = self._make_layer(output_channels,
                                     output_channels,
                                     groups=output_channels,
                                     number=value)
            setattr(self, f'layer{idx}', layer)

        self.CB = nn.Sequential(
            nn.Conv2d(output_channels * len(layers),
                      output_channels,
                      1,
                      1,
                      0,
                      bias=False),
            nn.BatchNorm2d(output_channels),
        )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)

        out = []
        for idx,_ in enumerate(self.layers):
            tmp = getattr(self, f'layer{idx}')(x)
            out.append(tmp)

        y = self.CB(torch.concat(out, dim=1))

        result = self.relu(x + y)
        return result

    def _make_layer(self,
                    inp,
                    oup,
                    kernel=5,
                    stride=1,
                    padding=2,
                    groups=1,
                    bias=False,
                    number=1):
        layer = []
        for _ in range(number):
            layer.append(
                CBR(inp,
                    oup,
                    kernel,
                    stride,
                    padding=padding,
                    bias=bias,
                    groups=groups))
            return nn.Sequential(*layer)
