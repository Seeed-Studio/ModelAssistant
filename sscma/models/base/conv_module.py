# copyright Copyright (c) Seeed Technology Co.,Ltd.
from .general import ConvNormActivation


class ConvModule:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv_cfg, norm_cfg, act_cfg):
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv = ConvNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_layer=conv_cfg,
            norm_layer=norm_cfg,
            activation_layer=act_cfg,
        )
        return self.conv
