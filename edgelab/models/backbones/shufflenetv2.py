import copy

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import VISBACKENDS
from mmpose.models.backbones.shufflenet_v2 import ShuffleNetV2


@VISBACKENDS.register_module(force=True)
class CustomShuffleNetV2(ShuffleNetV2):

    def __init__(self,
                 widen_factor=1.0,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False):
        # super().__init__(widen_factor, out_indices, frozen_stages, conv_cfg,
        #                  norm_cfg, act_cfg, norm_eval, with_cp)

        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 4). But received {index}')

        if frozen_stages not in range(-1, 4):
            raise ValueError('frozen_stages must be in range(-1, 4). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if widen_factor == 0.25:
            channels = [48, 96, 192]
        elif widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
        else:
            raise ValueError(
                'widen_factor must be in [0.25, 0.5, 1.0, 1.5, 2.0]. '
                f'But received {widen_factor}')

        self.in_channels = 24
        self.conv1 = ConvModule(in_channels=3,
                                out_channels=self.in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            layer = self._make_layer(channels[i], num_blocks)
            self.layers.append(layer)

        output_channels = channels[-1]
        self.layers.append(
            ConvModule(in_channels=self.in_channels,
                       out_channels=output_channels,
                       kernel_size=1,
                       conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg))
