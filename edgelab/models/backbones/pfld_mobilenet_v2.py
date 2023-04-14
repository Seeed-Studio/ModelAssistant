import torch.nn as nn
from mmpose.models.builder import BACKBONES

from ..base.general import InvertedResidual, CBR


@BACKBONES.register_module()
class PfldMobileNetV2(nn.Module):

    def __init__(self,
                 inchannel=3,
                 layer1=[16, 16, 16, 16, 16],
                 layer2=[32, 32, 32, 32, 32, 32],
                 out_channel=16):
        super(PfldMobileNetV2, self).__init__()
        inp = 32
        self.conv1 = CBR(inchannel,
                         inp,
                         kernel=3,
                         stride=2,
                         padding=1,
                         bias=False)
        self.conv2 = CBR(inp,
                         inp,
                         kernel=3,
                         stride=1,
                         padding=1,
                         groups=inp,
                         bias=False)

        layer = []
        for idx, oup in enumerate(layer1):
            if idx == 0:
                layer.append(InvertedResidual(inp, oup, 2, False, 2))
            else:
                layer.append(InvertedResidual(inp, oup, 1, True, 2))
            inp = oup
        self.layer1 = nn.Sequential(*layer)
        self.block1 = InvertedResidual(inp, 32, 2, False, 2)
        inp = 32

        layer = []
        for idx, oup in enumerate(layer2):
            layer.append(InvertedResidual(inp, oup, 1, True if idx else False, 4))
            inp = oup
        self.layer2 = nn.Sequential(*layer)
        self.block2 = InvertedResidual(inp, out_channel, 1, False, 2)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.layer1(x)

        x = self.block1(x)

        x = self.layer2(x)

        x = self.block2(x)

        return x
