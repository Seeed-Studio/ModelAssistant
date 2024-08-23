# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .rtmdet_l_8xb32_300e_coco import *

from sscma.datasets.transforms.loading import LoadImageFromFile
from sscma.datasets.transforms.processing import RandomResize
from mmengine.hooks.ema_hook import EMAHook
from mmengine.hooks.profiler_hook import ProfilerHook


from sscma.datasets.transforms.formatting import PackDetInputs
from sscma.datasets.transforms.loading import LoadAnnotations
from sscma.datasets.transforms.transforms import (
    MixUp,
    Mosaic,
    Pad,
    RandomCrop,
    RandomFlip,
    Resize,
    HSVRandomAug,
    toTensor,
)
from sscma.engine.hooks.pipeline_switch_hook import PipelineSwitchHook
from sscma.models.layers.ema import ExpMomentumEMA

d_factor = 0.33
w_factor = 0.25
imgsz = (320, 320)

max_epochs = 300
stage2_num_epochs = 20


model.update(
    dict(
        backbone=dict(
            deepen_factor=d_factor,
            widen_factor=w_factor,
            use_depthwise=True,
        ),
        neck=dict(
            deepen_factor=d_factor,
            widen_factor=w_factor,
            num_csp_blocks=1,
            use_depthwise=True,
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
    dict(type=toTensor),
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
    # dict(
    #     type=MixUp,
    #     img_scale=(input_shape, input_shape),
    #     ratio_range=(1.0, 1.0),
    #     max_cached_images=20,
    #     pad_val=114.0,
    # ), #removed MixUp
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
    dict(type=toTensor),
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
    dict(batch_size=128, num_workers=16, dataset=dict(pipeline=train_pipeline))
)


custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type=PipelineSwitchHook, switch_epoch=max_epochs - stage2_num_epochs, switch_pipeline=train_pipeline_stage2
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
