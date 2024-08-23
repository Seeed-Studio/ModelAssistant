# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .rtmdet_l_8xb32_300e_coco import *

deepen_factor = 0.67
widen_factor = 0.75

model.update(
    dict(
        backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
        neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
        bbox_head=dict(head_module=dict(widen_factor=widen_factor))
    )
)
