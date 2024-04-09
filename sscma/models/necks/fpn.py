# Copyright (c) Seeed Tech Ltd. All rights reserved.
from typing import List, Optional, Tuple, Union

from mmdet.models.necks.fpn import FPN as _FPN
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn import functional as F

from sscma.registry import NECKS


@NECKS.register_module()
class FPN(_FPN):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        out_idx: Tuple[int, ...] = [0],
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = True,
        no_norm_on_lateral: bool = False,
        conv_cfg: Optional[Union[ConfigDict, dict]] = None,
        norm_cfg: Optional[Union[ConfigDict, dict]] = None,
        act_cfg: Optional[Union[ConfigDict, dict]] = None,
        upsample_cfg: Optional[Union[ConfigDict, dict]] = dict(mode='nearest'),
        init_cfg: Union[Union[ConfigDict, dict], List[Union[ConfigDict, dict]]] = dict(
            type='Xavier', layer='Conv2d', distribution='uniform'
        ),
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            upsample_cfg,
            init_cfg,
        )
        self.out_idx = out_idx
        assert len(out_idx) <= num_outs

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_idx]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


@NECKS.register_module()
class LiteFPN(BaseModule):
    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
