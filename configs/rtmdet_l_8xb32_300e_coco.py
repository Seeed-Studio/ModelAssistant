# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *
    from .schedules.schedule_1x import *
    from .datasets.coco_detection import *

from torchvision.ops import nms
from sscma.datasets.transforms.loading import LoadImageFromFile
from sscma.datasets.transforms.processing import RandomResize
from mmengine.hooks.ema_hook import EMAHook
from mmengine.hooks.profiler_hook import  ProfilerHook
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.nn import SyncBatchNorm
from torch.nn.modules.activation import SiLU
from torch.optim.adamw import AdamW

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
from sscma.models.backbones.cspnext import CSPNeXt
from sscma.datasets.data_preprocessor import DetDataPreprocessor
from sscma.models.heads.rtmdet_head import RTMDetHead, RTMDetSepBNHeadModule
from sscma.models.detectors.rtmdet import RTMDet
from sscma.models.layers.ema import ExpMomentumEMA
from sscma.models.losses.gfocal_loss import QualityFocalLoss
from sscma.models.losses.iou_loss import GIoULoss
from sscma.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from sscma.models.task_modules.assigners.dynamic_soft_label_assigner import (
    DynamicSoftLabelAssigner,
)
from sscma.models.task_modules.assigners.batch_dsl_assigner import (
    BatchDynamicSoftLabelAssigner,
)
from sscma.models.task_modules.coders.distance_point_bbox_coder import (
    DistancePointBBoxCoder,
)
from sscma.models.task_modules.prior_generators.point_generator import (
    MlvlPointGenerator,
)
from sscma.models.task_modules.assigners.iou2d_calculator import BboxOverlaps2D

from sscma.engine.hooks.visualization_hook import DetVisualizationHook
from sscma.visualization.local_visualizer import DetLocalVisualizer


default_hooks.visualization = dict(type=DetVisualizationHook)

visualizer = dict(type=DetLocalVisualizer, vis_backends=vis_backends, name="visualizer")


model = dict(
    type=RTMDet,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    backbone=dict(
        type=CSPNeXt,
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type=SyncBatchNorm),
        act_cfg=dict(type=SiLU, inplace=True),
    ),
    neck=dict(
        type=CSPNeXtPAFPN,
        deepen_factor=1,
        widen_factor=1,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type=SyncBatchNorm),
        act_cfg=dict(type=SiLU, inplace=True),
    ),
    bbox_head=dict(
        type=RTMDetHead,
        head_module=dict(
            type=RTMDetSepBNHeadModule,
            num_classes=80,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            norm_cfg=dict(type=SyncBatchNorm),
            act_cfg=dict(type=SiLU, inplace=True),
            share_conv=True,
            pred_kernel_size=1,
            featmap_strides=[8, 16, 32],
        ),
        prior_generator=dict(type=MlvlPointGenerator, offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type=DistancePointBBoxCoder),
        loss_cls=dict(
            type=QualityFocalLoss, use_sigmoid=True, beta=2.0, loss_weight=1.0
        ),
        loss_bbox=dict(type=GIoULoss, loss_weight=2.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type=BatchDynamicSoftLabelAssigner,
            num_classes=80,
            topk=13,
            iou_calculator=dict(type=BboxOverlaps2D),
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type=nms, iou_threshold=0.65),
        max_per_img=300,
    ),
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
    dict(type=Mosaic, img_scale=(640, 640), pad_val=114.0),
    dict(
        type=RandomResize,
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=(640, 640)),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type=MixUp,
        img_scale=(640, 640),
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
    dict(type=toTensor),
    dict(
        type=RandomResize,
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=(640, 640)),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=toTensor),
    dict(type=Resize, scale=(640, 640), keep_ratio=True),
    dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]


train_dataloader.update(
    dict(
        batch_size=32,
        num_workers=12,
        batch_sampler=None,
        pin_memory=True,
        collate_fn=coco_collate,
        dataset=dict(pipeline=train_pipeline),
    )
)

val_dataloader.update(
    dict(batch_size=5, num_workers=8, dataset=dict(pipeline=test_pipeline))
)
test_dataloader = val_dataloader

max_epochs = 300
stage2_num_epochs = 20
base_lr = 0.004
interval = 5

train_cfg.update(
    dict(
        max_epochs=max_epochs,
        val_interval=interval,
        dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)],
    )
)

val_evaluator.update(dict(proposal_nums=(100, 1, 10)))
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type=CosineAnnealingLR,
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# hooks
default_hooks.update(
    dict(
        checkpoint=dict(
            interval=interval,
            max_keep_ckpts=3,  # only keep latest 3 checkpoints
        )
    )
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
        type=PipelineSwitchHook,
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    )
]
