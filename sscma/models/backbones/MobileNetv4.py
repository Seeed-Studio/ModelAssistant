from typing import List, Text, Dict, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor

from sscma.models.base.general import ConvNormActivation
from sscma.registry import MODELS
from sscma.models.layers.nn_blocks import (
    UniversalInvertedBottleneckBlock,
    InvertedBottleneckBlock,
    MultiHeadSelfAttentionBlock,
)


class BlockConfig:
    block_name: str = 'convnormact'
    input_channels: int = 3
    output_channels: int = 128
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    isoutputblock: bool = False
    expand_ratio: int = 1
    start_dw_kernel_size: int = 3
    middle_dw_kernel_size: int = 3
    middle_dw_downsample: bool = False
    bias: bool = False
    downsampling_dw_kernel_size: int = 3
    use_residual: bool = True

    def __init__(
        self,
        block_name: str,
        activation: str,
        kernel_size: Optional[int] = 3,
        start_dw_kernel_size: Optional[int] = 3,
        middle_dw_kernel_size: Optional[int] = 3,
        middle_dw_downsample: bool = False,
        stride: int = 1,
        output_channels: int = 128,
        expand_ratio: Optional[int] = 1,
        isoutputblock: bool = False,
        input_channels: int = 3,
        **kwargs,
    ) -> None:
        self.block_name = block_name
        self.activation = activation
        self.kernel_size = kernel_size
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size
        self.middle_dw_downsample = middle_dw_downsample
        self.stride = stride
        self.output_channels = output_channels
        self.isoutputblock = isoutputblock
        self.expand_ratio = expand_ratio
        self.input_channels = input_channels

        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)


def mhsa_medium_24px():
    return dict(
        block_name='mhsa',
        activation='relu',
        output_channels=160,
        key_dim=64,
        value_dim=64,
        query_h_strides=1,
        query_w_strides=1,
        kv_strides=2,
        num_heads=4,
        use_layer_scale=True,
        use_multi_query=True,
        isoutputblock=False,
    )


def mhsa_medium_12px():
    return dict(
        block_name='mhsa',
        activation='relu',
        output_channels=256,
        key_dim=64,
        value_dim=64,
        query_h_strides=1,
        query_w_strides=1,
        kv_strides=1,
        num_heads=4,
        use_layer_scale=True,
        use_multi_query=True,
        isoutputblock=False,
    )


def mhsa_large_24px():
    return dict(
        block_name='mhsa',
        activation='relu',
        output_channels=192,
        key_dim=48,
        value_dim=48,
        query_h_strides=1,
        query_w_strides=1,
        kv_strides=2,
        num_heads=8,
        use_layer_scale=True,
        use_multi_query=True,
        isoutputblock=False,
    )


def mhsa_large_12px():
    return dict(
        block_name='mhsa',
        activation='relu',
        output_channels=512,
        key_dim=64,
        value_dim=64,
        query_h_strides=1,
        query_w_strides=1,
        kv_strides=1,
        num_heads=8,
        use_layer_scale=True,
        use_multi_query=True,
        isoutputblock=False,
    )


@MODELS.register_module()
class MobileNetv4(nn.Module):
    '''
    Architecture: https://arxiv.org/abs/2404.10518

    "MobileNetV4 - Universal Models for the Mobile Ecosystem"
    Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan
    Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal,
    Tenghui Zhu, Daniele Moro, Andrew Howard
    '''

    Arch = {
        'small': [
            ('convbn', 'ReLU', 3, None, None, False, 2, 32, None, False),  # 1/2
            ('fused_ib', 'ReLU', 3, None, None, False, 2, 32, 1, False),  # 1/4
            ('fused_ib', 'ReLU', 3, None, None, False, 2, 64, 3, True),  # 1/8
            ('uib', 'ReLU', None, 5, 5, True, 2, 96, 3.0, False),  # 1/16
            ('uib', 'ReLU', None, 0, 3, True, 1, 96, 2.0, False),  # IB
            ('uib', 'ReLU', None, 0, 3, True, 1, 96, 2.0, False),  # IB
            ('uib', 'ReLU', None, 0, 3, True, 1, 96, 2.0, False),  # IB
            ('uib', 'ReLU', None, 0, 3, True, 1, 96, 2.0, False),  # IB
            ('uib', 'ReLU', None, 3, 0, True, 1, 96, 4.0, True),  # ConvNext
            ('uib', 'ReLU', None, 3, 3, True, 2, 128, 6.0, False),  # 1/32
            ('uib', 'ReLU', None, 5, 5, True, 1, 128, 4.0, False),  # ExtraDW
            ('uib', 'ReLU', None, 0, 5, True, 1, 128, 4.0, False),  # IB
            ('uib', 'ReLU', None, 0, 5, True, 1, 128, 3.0, False),  # IB
            ('uib', 'ReLU', None, 0, 3, True, 1, 128, 4.0, False),  # IB
            ('uib', 'ReLU', None, 0, 3, True, 1, 128, 4.0, True),  # IB
            ('convbn', 'ReLU', 1, None, None, False, 1, 960, None, False),
            ('avgpool', None, 7, None, None, None, None, None, None, False),  # Avg
            ('line', 'ReLU', 1, None, None, False, 1, 1280, None, False),  # Conv
            ('line', 'ReLU', 1, None, None, False, 1, 1000, None, False),  # Conv
        ],
        'medium': [
            ('convbn', 'ReLU', 3, None, None, False, 2, 32, None, False),  # 1/2
            ('fused_ib', 'ReLU', 3, None, None, False, 2, 48, 3, False),  # 1/4
            ('uib', 'ReLU', None, 3, 5, True, 2, 80, 2.0, False),  # IB
            ('uib', 'ReLU', None, 3, 3, True, 1, 80, 2.0, True),  # IB 1/8
            ('uib', 'ReLU', None, 3, 5, True, 2, 160, 3.0, False),  # IB
            ('uib', 'ReLU', None, 3, 3, True, 1, 160, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 3, True, 1, 160, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 5, True, 1, 160, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 3, True, 1, 160, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 0, True, 1, 160, 4.0, False),  # IB
            ('uib', 'ReLU', None, 0, 0, True, 1, 160, 2.0, False),  # IB
            ('uib', 'ReLU', None, 3, 0, True, 1, 160, 4.0, True),  # IB
            ('uib', 'ReLU', None, 5, 5, True, 2, 256, 4.0, False),  # IB
            ('uib', 'ReLU', None, 5, 5, True, 1, 256, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 5, True, 1, 256, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 5, True, 1, 256, 4.0, False),  # IB
            ('uib', 'ReLU', None, 0, 0, True, 1, 256, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 0, True, 1, 256, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 5, True, 1, 256, 2.0, False),  # IB
            ('uib', 'ReLU', None, 5, 5, True, 1, 256, 4.0, False),  # IB
            ('uib', 'ReLU', None, 0, 0, True, 1, 256, 4.0, False),  # IB
            ('uib', 'ReLU', None, 0, 0, True, 1, 256, 4.0, False),  # IB
            ('uib', 'ReLU', None, 5, 0, True, 1, 256, 2.0, True),  # IB
            ('convbn', 'ReLU', 1, None, None, False, 1, 960, None, False),  # 1/2
            ('avgpool', None, 8, None, None, None, None, None, None, False),
            ('line', 'ReLU', 1, None, None, False, 1, 1280, None, False),
            ('line', 'ReLU', 1, None, None, False, 1, 1000, None, False),
        ],
        'large': [
            ('convbn', 'ReLU', 3, None, None, False, 2, 24, None, False),
            ('fused_ib', 'ReLU', 3, None, None, False, 2, 48, 4.0, True),
            ('uib', 'ReLU', None, 3, 5, True, 2, 96, 4.0, False),
            ('uib', 'ReLU', None, 3, 3, True, 1, 96, 4.0, True),
            ('uib', 'ReLU', None, 3, 5, True, 2, 192, 4.0, False),
            ('uib', 'ReLU', None, 3, 3, True, 1, 192, 4.0, False),
            ('uib', 'ReLU', None, 3, 3, True, 1, 192, 4.0, False),
            ('uib', 'ReLU', None, 3, 3, True, 1, 192, 4.0, False),
            ('uib', 'ReLU', None, 3, 5, True, 1, 192, 4.0, False),
            ('uib', 'ReLU', None, 5, 3, True, 1, 192, 4.0, False),
            ('uib', 'ReLU', None, 5, 3, True, 1, 192, 4.0, False),
            ('uib', 'ReLU', None, 5, 3, True, 1, 192, 4.0, False),
            ('uib', 'ReLU', None, 5, 3, True, 1, 192, 4.0, False),
            ('uib', 'ReLU', None, 5, 3, True, 1, 192, 4.0, False),
            ('uib', 'ReLU', None, 3, 0, True, 1, 192, 4.0, True),
            ('uib', 'ReLU', None, 5, 5, True, 2, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 5, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 5, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 5, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 0, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 3, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 0, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 0, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 3, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 5, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 0, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 0, True, 1, 512, 4.0, False),
            ('uib', 'ReLU', None, 5, 0, True, 1, 512, 4.0, True),
            ('convbn', 'ReLU', 1, None, None, False, 1, 960, None, False),
            ('avgpool', None, None, None, None, None, None, None, None, False),
            ('line', 'ReLU', 1, None, None, False, 1, 1280, None, False),
            ('line', 'ReLU', 1, None, None, False, 1, 1000, None, False),
        ],
        'hybridmedium': [
            ('convbn', 'ReLU', 3, None, None, False, 2, 32, None, False),  # 1/2
            ('fused_ib', 'ReLU', 3, None, None, False, 2, 48, 4, True),  # 1/4
            ('uib', 'ReLU', None, 3, 5, True, 2, 80, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 3, True, 1, 80, 2.0, False),  # IB
            ('uib', 'ReLU', None, 3, 5, True, 2, 160, 6.0, False),  # IB
            ('uib', 'ReLU', None, 0, 0, True, 1, 160, 2.0, False),  # IB
            ('uib', 'ReLU', None, 3, 3, True, 1, 160, 4.0, False),  # IB
            ('uib', 'ReLU', None, 3, 5, True, 1, 160, 4.0, False),  # IB
            mhsa_medium_24px(),
            ('uib', 'ReLU', None, 3, 3, True, 1, 160, 4.0, False),  # IB
            mhsa_medium_24px(),
            ('uib', 'ReLU', None, 3, 0, True, 1, 160, 4.0, False),  # IB
            mhsa_medium_24px(),
            ('uib', 'ReLU', None, 3, 3, True, 1, 160, 4.0, False),
            mhsa_medium_24px(),
            ('uib', 'ReLU', None, 3, 0, True, 1, 160, 4.0, True),
            ('uib', 'ReLU', None, 5, 5, True, 2, 256, 6.0, True),
            ('uib', 'ReLU', None, 5, 5, True, 1, 256, 4.0, False),
            ('uib', 'ReLU', None, 3, 5, True, 1, 256, 4.0, False),
            ('uib', 'ReLU', None, 3, 5, True, 1, 256, 4.0, False),
            ('uib', 'ReLU', None, 0, 0, True, 1, 256, 2.0, False),
            ('uib', 'ReLU', None, 3, 5, True, 1, 256, 2.0, False),
            ('uib', 'ReLU', None, 0, 0, True, 1, 256, 2.0, False),
            ('uib', 'ReLU', None, 0, 0, True, 1, 256, 4.0, False),
            mhsa_medium_12px(),
            ('uib', 'ReLU', None, 3, 0, True, 1, 256, 4.0, False),
            mhsa_medium_12px(),
            ('uib', 'ReLU', None, 5, 5, True, 1, 256, 4.0, False),
            mhsa_medium_12px(),
            ('uib', 'ReLU', None, 5, 0, True, 1, 256, 4.0, False),
            mhsa_medium_12px(),
            ('uib', 'ReLU', None, 0, 0, True, 1, 256, 4.0, True),
            ('convbn', 'ReLU', 1, None, None, False, 1, 960, None, False),
            ('avgpool', None, None, None, None, None, None, None, None, False),
            ('line', 'ReLU', 1, None, None, False, 1, 1280, None, False),
            ('line', 'ReLU', 1, None, None, False, 1, 1000, None, False),
        ],
        'hybridlarge': [
            ('convbn', 'GELU', 3, None, None, False, 2, 24, None, False),  # 1/2
            ('fused_ib', 'GELU', 3, None, None, False, 2, 48, 4, True),  # 1/4
            ('uib', 'GELU', None, 3, 5, True, 2, 96, 4.0, False),  # IB
            ('uib', 'GELU', None, 3, 3, True, 1, 96, 4.0, True),  # IB
            ('uib', 'GELU', None, 3, 5, True, 2, 192, 4.0, False),  # IB
            ('uib', 'GELU', None, 3, 3, True, 1, 192, 4.0, False),  # IB
            ('uib', 'GELU', None, 3, 3, True, 1, 192, 4.0, False),  # IB
            ('uib', 'GELU', None, 3, 3, True, 1, 192, 4.0, False),  # IB
            ('uib', 'GELU', None, 3, 5, True, 1, 192, 4.0, False),  # IB
            ('uib', 'GELU', None, 5, 3, True, 1, 192, 4.0, False),  # IB
            ('uib', 'GELU', None, 5, 3, True, 1, 192, 4.0, False),  # IB
            mhsa_large_24px(),
            ('uib', 'GELU', None, 5, 3, True, 1, 192, 4.0, False),  # IB
            mhsa_large_24px(),
            ('uib', 'GELU', None, 5, 3, True, 1, 192, 4.0, False),  # IB
            mhsa_large_24px(),
            ('uib', 'GELU', None, 5, 3, True, 1, 192, 4.0, False),
            mhsa_large_24px(),
            ('uib', 'GELU', None, 3, 0, True, 1, 192, 4.0, True),  # output
            ('uib', 'GELU', None, 5, 5, True, 2, 512, 4.0, True),
            ('uib', 'GELU', None, 5, 5, True, 1, 512, 4.0, False),
            ('uib', 'GELU', None, 5, 5, True, 1, 512, 4.0, False),
            ('uib', 'GELU', None, 5, 5, True, 1, 512, 4.0, False),
            ('uib', 'GELU', None, 5, 0, True, 1, 512, 4.0, False),
            ('uib', 'GELU', None, 5, 3, True, 1, 512, 4.0, False),
            ('uib', 'GELU', None, 5, 0, True, 1, 512, 4.0, False),
            ('uib', 'GELU', None, 5, 0, True, 1, 512, 4.0, False),
            ('uib', 'GELU', None, 5, 3, True, 1, 512, 4.0, False),
            ('uib', 'GELU', None, 5, 5, True, 1, 512, 4.0, False),
            mhsa_large_12px(),
            ('uib', 'ReLU', None, 5, 0, True, 1, 512, 4.0, False),
            mhsa_large_12px(),
            ('uib', 'GELU', None, 5, 0, True, 1, 512, 4.0, False),
            mhsa_large_12px(),
            ('uib', 'GELU', None, 5, 0, True, 1, 512, 4.0, False),
            mhsa_large_12px(),
            ('uib', 'GELU', None, 5, 0, True, 1, 512, 4.0, True),
            ('convbn', 'GELU', 1, None, None, False, 1, 960, None, False),
            ('avgpool', None, None, None, None, None, None, None, None, False),
            ('line', 'GELU', 1, None, None, False, 1, 1280, None, False),
            ('line', 'GELU', 1, None, None, False, 1, 1000, None, False),
        ],
    }

    def __init__(
        self,
        arch: str,
        input_channels=3,
        stochastic_depth_drop_rate: float = 0.0,
        use_sync_bn: bool = False,
        output_intermediate_endpoints: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        arch_setting = self.Arch[arch]
        self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self._use_sync_bn = use_sync_bn
        self._output_intermediate_endpoints = output_intermediate_endpoints

        self._output_stride: int = (1,)

        self.block_settings = []
        for setting in arch_setting:
            if isinstance(setting, tuple):
                self.block_settings.append(BlockConfig(*setting, input_channels=input_channels))
            else:
                self.block_settings.append(BlockConfig(**setting, input_channels=input_channels))
            if self.block_settings[-1].output_channels is not None:
                input_channels = self.block_settings[-1].output_channels

        last_output_block = 0
        for i, block in enumerate(self.block_settings):
            if block.isoutputblock:
                last_output_block = i

        self._forward_blocks = self.build_layers()[: last_output_block + 1]

    def build_layers(self):
        layers = []
        block: BlockConfig
        current_stride = 1
        rate = 1
        for block in self.block_settings:

            if not block.stride:
                block.stride = 1

            if self._output_stride is not None and current_stride == self._output_stride:
                layer_stride = 1
                layer_rate = layer_rate
                rate *= block.stride
            else:
                layer_stride = block.stride
                layer_rate = 1
                current_stride *= block.stride

            if block.block_name == 'convbn':
                layer = ConvNormActivation(
                    block.input_channels,
                    block.output_channels,
                    kernel_size=block.kernel_size,
                    stride=block.stride,
                    activation_layer=block.activation,
                    norm_layer=dict(type='BN', eps=0.001, momentum=0.99),
                    bias=block.bias,
                )
            elif block.block_name == 'uib':
                layer = UniversalInvertedBottleneckBlock(
                    block.input_channels,
                    block.output_channels,
                    block.expand_ratio,
                    stride=layer_stride,
                    activation=block.activation,
                    dialation=1,
                    start_dw_kernel_size=block.start_dw_kernel_size,
                    middle_dw_kernel_size=block.middle_dw_kernel_size,
                    middle_dw_downsample=block.middle_dw_downsample,
                )
            elif block.block_name == 'line':
                layer = nn.Sequential(nn.Flatten(), nn.Linear(block.input_channels, block.output_channels), nn.ReLU())
            elif block.block_name == 'mhsa':
                layer = MultiHeadSelfAttentionBlock(
                    in_channels=block.input_channels,
                    out_channels=block.output_channels,
                    num_heads=block.num_heads,
                    key_dim=block.key_dim,
                    value_dim=block.value_dim,
                    use_multi_query=block.use_multi_query,
                    query_h_strides=block.query_h_strides,
                    query_w_strides=block.query_w_strides,
                    kv_strides=block.kv_strides,
                    downsampling_dw_kernel_size=block.downsampling_dw_kernel_size,
                    cpe_dw_kernel_size=block.kernel_size,
                    stochastic_depth_drop_rate=self._stochastic_depth_drop_rate,
                    use_residual=block.use_residual,
                    use_sync_bn=self._use_sync_bn,
                    use_layer_scale=block.use_layer_scale,
                    output_intermediate_endpoints=self._output_intermediate_endpoints,
                )
            elif block.block_name == 'avgpool':
                if block.kernel_size is not None:
                    layer = nn.AvgPool2d(kernel_size=block.kernel_size, stride=1, padding=0)
                else:
                    layer = nn.AdaptiveAvgPool2d(1)
            elif block.block_name in ('fused_ib', 'inverted_bottleneck'):
                layer = InvertedBottleneckBlock(
                    block.input_channels,
                    block.output_channels,
                    block.expand_ratio,
                    layer_stride,
                    block.kernel_size,
                    dilation=rate,
                    activation=block.activation,
                    use_residual=True,
                    expand_se_in_filters=True,
                    use_depthwise=True,
                    division=8,
                    output_intermediate_endpoints=False,
                    squeeze_ratio=None,
                    se_start_activation=nn.Hardsigmoid,
                )
            else:
                raise ValueError(f'block name "{block.block_name}" is not supported')

            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        for cfg, blk in zip(self.block_settings, self._forward_blocks):
            x = blk(x)
            if cfg.isoutputblock:
                outs.append(x)

        return tuple(outs)
