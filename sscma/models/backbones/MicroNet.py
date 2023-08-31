from typing import Optional, Union, Tuple, Dict, Callable
from mmcls.models.classifiers.base import BaseClassifier
from mmengine.model.base_module import BaseModule
import torch.nn as nn

from sscma.registry import MODELS
from sscma.models.base.general import ConvNormActivation
from sscma.models.layers import RepConv1x1


class MicroBlock(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        normal_cfg=None,
        act_cfg=None,
    ) -> None:
        super().__init__()

        self.conv1 = ConvNormActivation(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=1,
            padding=1 if kernel == 3 else 0,
            norm_layer=normal_cfg,
        )
        self.conv2 = ConvNormActivation(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=out_channels,
            activation_layer=act_cfg,
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


@MODELS.register_module()
class MicroNet(BaseClassifier):
    """
    paper: https://arxiv.org/pdf/2010.11267.pdf

    """

    archs = {
        # c,k,s
        "s": [[72, 3, 2], [164, 1, 1], [220, 1, 2], [276, 1, 2]],
        "m": [
            [192, 3, 2],
            [276, 1, 1],
            [276, 1, 1],
            [276, 1, 2],
            [276, 1, 2],
        ],
        "l": [
            [276, 3, 2],
            [248, 1, 1],
            [276, 1, 1],
            [276, 1, 2],
            [248, 1, 2],
        ],
    }

    def __init__(
        self,
        arch: str = 's',
        gray: bool = False,
        rep: bool = False,
        out_indices: Union[int, Tuple[int]] = (-1,),
        act_cfg: Union[Dict, str, Callable] = "ReLU6",
        init_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg, data_preprocessor)
        if arch not in self.archs.keys():
            raise ValueError(f'The archc parameter must be one of "s", "m", "l", but received "{arch}"')
        arch = self.archs[arch]
        self.out_indices = (out_indices,) if isinstance(out_indices, int) else out_indices
        if min(self.out_indices) > len(arch):
            print(
                f"Warning!!! The parameter set by the parameter {out_indices} is greater than the depth of the model,",
                f" which has been set to {len(arch)} (the maximum depth of the model) by default, but it may cause unexpected errors.",
            )
            self.out_indices = (len(arch),)
        elif min(self.out_indices) < 0:
            self.out_indices = (len(arch),)

        in_channels = 1 if gray else 3

        self.layers = nn.ModuleList()
        for out_channels, kernel, stride in arch:
            if rep:
                layer = RepConv1x1(
                    in_channels,
                    out_channels,
                    stride=stride,
                    act_cfg=act_cfg,
                )
            else:
                layer = MicroBlock(in_channels, out_channels, kernel=kernel, stride=stride, act_cfg=act_cfg)
            in_channels = out_channels
            self.layers.append(layer)
        end = ConvNormActivation(in_channels, in_channels, 1, 1, 0, bias=True, activation_layer=act_cfg)
        self.layers.append(end)

    def forward(self, x) -> tuple:
        res = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.out_indices:
                res.append(x)
                if idx == max(self.out_indices):
                    break

        return tuple(res)

    def init_weights(self) -> None:
        super().init_weights()
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
