# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
"""Swift-YOLO (YOLOv5-style) tiny on COCO - main-branch config.

Ported from the 2.0.0 branch (configs/swift_yolo). On main there is no
mmcv/mmdet: all building blocks are vendored in sscma and referenced as
Python classes directly.
"""

from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *

from mmengine.hooks import EMAHook, CheckpointHook
from mmengine.optim import OptimWrapper
from torch.optim import SGD
from torch.nn import ReLU

from sscma.datasets import CustomYOLOv5CocoDataset, DetDataPreprocessor
from sscma.datasets.coco import BatchShapePolicy
from sscma.datasets.transforms import (
    Albu,
    LetterResize,
    LoadAnnotations,
    LoadImageFromFile,
    Mosaic,
    PackDetInputs,
    RandomFlip,
    YOLOv5HSVRandomAug,
    YOLOv5KeepRatioResize,
    YOLOv5RandomAffine,
)
from sscma.engine import YOLOv5OptimizerConstructor, YOLOv5ParamSchedulerHook
from sscma.deploy.models.yolo_infer import YOLOInfer
from sscma.evaluation import CocoMetric
from sscma.models import (
    CrossEntropyLoss,
    DetHead,
    YOLOv5IoULoss,
    YOLOAnchorGenerator,
    YOLODetector,
    YOLOV5Head,
    YOLOv5CSPDarknet,
    YOLOv5PAFPN,
)

# ========================Suggested optional parameters========================
# MODEL
num_classes = 71
deepen_factor = 0.33
widen_factor = 0.15

# DATA
dataset_type = CustomYOLOv5CocoDataset
train_ann = 'train/_annotations.coco.json'
train_data = 'train/'  # Prefix of train image path
val_ann = 'valid/_annotations.coco.json'
val_data = 'valid/'  # Prefix of val image path

# dataset link: https://universe.roboflow.com/team-roboflow/coco-128
# NOTE: ann_file for the evaluator is built as data_root + val_ann, so a
# local data_root must end with a path separator.
data_root = 'https://universe.roboflow.com/ds/z5UOcgxZzD?key=bwx9LQUT0t'
height = 640
width = 640
batch = 16
workers = 2
val_batch = 1
val_workers = 1
imgsz = (width, height)

# TRAIN
lr = 0.01
epochs = 300
weight_decay = 0.0005
momentum = 0.937
lr_factor = 0.01
# persistent_workers must be False if workers is 0
persistent_workers = True
# Save model checkpoint and validation intervals
val_interval = 5
save_interval = val_interval
# The maximum checkpoints to keep
max_keep_ckpts = 3

# ================================END=================================

# DATA
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio

# MODEL
strides = [8, 16, 32]
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)],  # P5/32
]
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
prior_match_thr = 4.0  # Prior box matching threshold
# The obj loss weights of the three output layers
obj_level_weights = [4.0, 1.0, 0.4]

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300,
)  # Max number of detections of each image

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type=BatchShapePolicy,
    batch_size=val_batch,
    img_size=imgsz[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5,
)

model = dict(
    type=YOLODetector,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type=YOLOv5CSPDarknet,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    neck=dict(
        type=YOLOv5PAFPN,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    bbox_head=dict(
        type=YOLOV5Head,
        head_module=dict(
            type=DetHead,
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3,
        ),
        prior_generator=dict(type=YOLOAnchorGenerator, base_sizes=anchors, strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight * (num_classes / 80 * 3 / num_det_layers),
        ),
        loss_bbox=dict(
            type=YOLOv5IoULoss,
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True,
        ),
        loss_obj=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight * ((imgsz[0] / 640) ** 2 * 3 / num_det_layers),
        ),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights,
    ),
    test_cfg=model_test_cfg,
)

deploy = dict(
    type=YOLOInfer,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
)

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01),
]

pre_transform = [
    dict(type=LoadImageFromFile, file_client_args=dict(backend='disk')),
    dict(type=LoadAnnotations, with_bbox=True),
]

train_pipeline = [
    *pre_transform,
    dict(type=Mosaic, img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type=YOLOv5RandomAffine,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # imgsz is (width, height)
        border=(-imgsz[0] // 2, -imgsz[1] // 2),
        border_val=(114, 114, 114),
    ),
    dict(
        type=Albu,
        transforms=albu_train_transforms,
        bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
    ),
    dict(type=YOLOv5HSVRandomAug),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=PackDetInputs, meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction')
    ),
]

train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=train_data),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    ),
)

test_pipeline = [
    dict(type=LoadImageFromFile, file_client_args=dict(backend='disk')),
    dict(type=YOLOv5KeepRatioResize, scale=imgsz),
    dict(type=LetterResize, scale=imgsz, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param'),
    ),
]

val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data),
        ann_file=val_ann,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg,
    ),
)

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=SGD, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True, batch_size_per_gpu=batch
    ),
    constructor=YOLOv5OptimizerConstructor,
)

default_hooks = dict(
    param_scheduler=dict(
        type=YOLOv5ParamSchedulerHook, scheduler_type='linear', lr_factor=lr_factor, max_epochs=epochs
    ),
    checkpoint=dict(type=CheckpointHook, interval=val_interval, save_best='auto', max_keep_ckpts=max_keep_ckpts),
)

custom_hooks = [
    dict(
        type=EMAHook, ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49
    )
]

val_evaluator = dict(type=CocoMetric, proposal_nums=(100, 1, 10), ann_file=data_root + val_ann, metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
