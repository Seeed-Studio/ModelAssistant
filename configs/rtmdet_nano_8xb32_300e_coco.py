# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from configs.rtmdet_l_8xb32_300e_coco import *

from mmengine.hooks import EMAHook
from sscma.datasets.transforms import (
    Mosaic,
    Pad,
    RandomCrop,
    RandomFlip,
    Resize,
    HSVRandomAug,
    LoadAnnotations,
    PackDetInputs,
    RandomResize,
    LoadImageFromFile,
)

from sscma.engine import PipelineSwitchHook
from sscma.models import ExpMomentumEMA

d_factor = 0.33
w_factor = 0.25

imgsz = (320, 320)

max_epochs = 120
stage2_num_epochs = 50


model.update(
    dict(
        backbone=dict(
            deepen_factor=d_factor,
            widen_factor=w_factor,
            use_depthwise=False,
        ),
        neck=dict(
            deepen_factor=d_factor,
            widen_factor=w_factor,
            num_csp_blocks=1,
            use_depthwise=False,
        ),
        bbox_head=dict(head_module=dict(widen_factor=w_factor, share_conv=False)),
    )
)

train_pipeline = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend="pillow",
        backend_args=None,
    ),
    dict(type=LoadAnnotations, imdecode_backend="pillow", with_bbox=True),
    dict(type=HSVRandomAug),
    dict(type=Mosaic, img_scale=imgsz, pad_val=114.0),
    dict(
        type=RandomResize,
        scale=(imgsz[0] * 2, imgsz[1] * 2),
        ratio_range=(0.5, 1.5),  # note: changed from 0.1 to 0.5, 2.0 to 1.5
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs),
]

train_pipeline_stage2 = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend="pillow",
        backend_args=None,
    ),
    dict(type=LoadAnnotations, imdecode_backend="pillow", with_bbox=True),
    dict(type=HSVRandomAug),
    dict(
        type=RandomResize,
        scale=(imgsz[0] * 2, imgsz[1] * 2),
        ratio_range=(0.5, 1.5),  # note: changed from 0.1 to 0.5, 2.0 to 1.5
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs),
]

train_dataloader.update(
    dict(batch_size=batch_size, num_workers=num_workers, dataset=dict(pipeline=train_pipeline))
)

dump_config = True
custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type=PipelineSwitchHook,
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
    # dict(
    #     type=ProfilerHook,
    #     activity_with_cpu=True,
    #     activity_with_cuda=True,
    #     profile_memory=True,
    #     record_shapes=True,
    #     with_stack=True,
    #     with_flops=True,
    #     schedule=dict(wait=1, warmup=1, active=2, repeat=1),
    #     on_trace_ready=dict(type="tb_trace"),
    # ),
]
