# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from mmengine.model import BaseModule
from ..cnn import ConvModule, DepthwiseSeparableConvModule
from ..layers import CSPLayer


class Transformation_Module(nn.Module):
    """Tranform channel dimension of backbone to identical number

    Args:
    """

    def __init__(
        self,
        in_channels=[116, 232, 464],
        out_channels=96,
        act_cfg="leaky_relu",
        conv_cfg=None,
        norm_cfg=None,
    ):
        super(Transformation_Module, self).__init__()

        self.trans = nn.ModuleList()
        for i in range(len(in_channels)):
            self.trans.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
            )

    def forward(self, x):

        outs = [self.trans[i](x[i]) for i in range(len(x))]

        return outs


class CSPPAN(BaseModule):
    """Path Aggregation Network with CSP module.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_features (int): Number of output features of CSPPAN module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        num_features=3,
        expansion=0.5,
        num_csp_blocks=1,
        use_depthwise=True,
        upsample_cfg=dict(scale_factor=2, mode="nearest"),
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="LeakyReLU"),
        spatial_scales=[0.125, 0.0625, 0.03125],
    ):

        super(CSPPAN, self).__init__()

        self.trans = Transformation_Module(
            in_channels, out_channels, act_cfg, conv_cfg, norm_cfg
        )
        in_channels = [out_channels] * len(spatial_scales)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_scales = spatial_scales
        self.num_features = num_features
        self.expansion = expansion

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        if self.num_features == 4:
            self.first_top_conv = conv(
                in_channels[0],
                in_channels[0],
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.second_top_conv = conv(
                in_channels[0],
                in_channels[0],
                kernel_size,
                padding=(kernel_size - 1) // 2,
                stride=2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            self.spatial_scales.append(self.spatial_scales[-1] / 2)

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            # recude已经通过transformation统一做过了。
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    kernel_size=kernel_size,
                    expansion=self.expansion,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    kernel_size,
                    stride=2,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    kernel_size=kernel_size,
                    expansion=self.expansion,
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: CSPPAN features.
        """

        assert len(inputs) == len(self.in_channels)
        inputs = self.trans(inputs)
        # top_down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            upsample_feat = self.upsample(feat_heigh)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1)
            )
            outs.append(out)

        top_features = None
        if self.num_features == 4:
            top_features = self.first_top_conv(inputs[-1])
            top_features = top_features + self.second_top_conv(outs[-1])
            outs.append(top_features)
        return tuple(outs)
