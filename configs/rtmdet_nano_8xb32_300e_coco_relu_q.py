# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .rtmdet_nano_8xb32_300e_coco_relu import *

from sscma.datasets.transforms.loading import LoadImageFromFile
from sscma.datasets.transforms.processing import RandomResize
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, ConstantLR


from sscma.datasets.transforms.formatting import PackDetInputs
from sscma.datasets.transforms.loading import LoadAnnotations
from sscma.datasets.transforms.transforms import (
    Pad,
    RandomCrop,
    RandomFlip,
    Resize,
    HSVRandomAug,
)
from sscma.engine.schedulers import QuadraticWarmupLR
from sscma.engine.hooks import QuantizerSwitchHook
from sscma.quantizer import RtmdetQuantModel


imgsz = (640, 640)
dump_config = False


max_epochs = 5
num_last_epochs = 2
base_lr = 0.00002

train_pipeline = [
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
model.bbox_head.update(train_cfg=model.train_cfg)
model.bbox_head.update(test_cfg=model.test_cfg)
quantizer_config = dict(
    type=RtmdetQuantModel,
    bbox_head=model.bbox_head,
    data_preprocessor=model.data_preprocessor,
)

train_dataloader.update(
    dict(batch_size=32, num_workers=16, dataset=dict(pipeline=train_pipeline))
)

train_cfg.update(
    dict(
        type=EpochBasedTrainLoop,
        max_epochs=max_epochs,
        val_interval=1,
        val_begin=1,
        dynamic_intervals=None,
    )
)

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.0005),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 3 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type=QuadraticWarmupLR,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True,
    ),
    dict(
        # use cosine lr from 5 to 35 epoch
        type=CosineAnnealingLR,
        eta_min=base_lr * 0.05,
        begin=1,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        # use fixed lr during last num_last_epochs epochs
        type=ConstantLR,
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    ),
]

custom_hooks = [
    dict(
        type=QuantizerSwitchHook,
        freeze_quantizer_epoch=max_epochs // 3,
        freeze_bn_epoch=max_epochs // 3 * 2,
    ),
]
