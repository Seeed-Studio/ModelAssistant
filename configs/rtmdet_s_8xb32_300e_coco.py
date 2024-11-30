# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .rtmdet_l_8xb32_300e_coco import *

from mmengine.hooks import ProfilerHook, EMAHook
from sscma.datasets.transforms import (
    MixUp,
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
w_factor = 0.5
imgsz = (640, 640)
max_epochs = 300
stage2_num_epochs = 20

checkpoint = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth"  # noqa
model.update(
    dict(
        backbone=dict(
            deepen_factor=d_factor,
            widen_factor=w_factor,
            init_cfg=dict(type="Pretrained", prefix="backbone.", checkpoint=checkpoint),
        ),
        #        neck=dict(in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
        neck=dict(
            deepen_factor=d_factor,
            widen_factor=w_factor,
        ),
        bbox_head=dict(head_module=dict(widen_factor=w_factor)),
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
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),  # note: changed from 0.1 to 0.5
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(
        type=MixUp,
        img_scale=imgsz,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=114.0,
    ),
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
        ratio_range=(0.5, 2.0),  # note: changed from 0.1 to 0.5
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs),
]

train_dataloader.update(dict(dataset=dict(pipeline=train_pipeline)))

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
    dict(
        type=ProfilerHook,
        activity_with_cpu=True,
        activity_with_cuda=True,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        schedule=dict(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=dict(type="tb_trace"),
    ),
]
