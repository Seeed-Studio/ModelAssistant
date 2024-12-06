# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_1x import *

from sscma.datasets import CustomFomoCocoDataset
from sscma.deploy.models.fomo_infer import FomoInfer
from mmengine.optim import LinearLR, MultiStepLR


from sscma.datasets.transforms import (
    Resize,
    LoadAnnotations,
    PackDetInputs,
    LoadImageFromFile,
    Bbox2FomoMask,
    Pad,
    HSVRandomAug,
    RandomFlip,
    Mosaic,
)
from sscma.engine import DetVisualizationHook
from sscma.visualization import FomoLocalVisualizer
from sscma.datasets import DetDataPreprocessor

from sscma.models import Fomo, MobileNetv2, FomoHead
from torch.nn import ReLU6, BCEWithLogitsLoss, ReLU

from torch.optim import Adam, SGD
from sscma.evaluation import FomoMetric

# ========================Suggested optional parameters========================
# MODEL
num_classes = 2
widen_factor = 0.35
# DATA
dataset_type = CustomFomoCocoDataset
# datasets link: https://public.roboflow.com/object-detection/mask-wearing
data_root = "/home/dq/code/sscma/datasets/hES8s8Gy7u"

train_ann = "train/_annotations.coco.json"
train_data = "train/"
val_ann = "valid/_annotations.coco.json"
val_data = "valid/"
metainfo = None

height = 192
width = 192
imgsz = (width, height)

downsample_factor = (8,)

# TRAIN
batch = 8
workers = 2
persistent_workers = True

val_batch = 16
val_workers = 2

lr = 0.02
epochs = 100

weight_decay = 0.0005
momentum = 0.95
# momentum = (0.9,0.99)

# ================================END=================================


default_hooks = dict(visualization=dict(type=DetVisualizationHook, score_thr=0.8))

visualizer = dict(type=FomoLocalVisualizer, fomo=True)


data_preprocessor = dict(
    type=DetDataPreprocessor,
    mean=[0, 0, 0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    pad_size_divisor=32,
)


model = dict(
    type=Fomo,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=MobileNetv2, widen_factor=widen_factor, out_indices=(2,), rep=False
    ),
    head=dict(
        type=FomoHead,
        input_channels=[16],
        num_classes=num_classes,
        middle_channel=48,
        act_cfg=ReLU,
        cls_weight=100.0,
        loss_cls=dict(type=BCEWithLogitsLoss, reduction="none"),
        loss_bg=dict(type=BCEWithLogitsLoss, reduction="none"),
    ),
    skip_preprocessor=True,
)

deploy = dict(
    type=FomoInfer,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
)

imdecode_backend = "cv2"

pre_transform = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend=imdecode_backend,
        file_client_args=dict(backend="disk"),
    ),
    dict(type=LoadAnnotations, with_bbox=True),
]

train_pipeline = [
    *pre_transform,
    dict(type=HSVRandomAug),
    dict(
        type=Mosaic,
        img_scale=imgsz,
        use_cached=True,
        #  max_cached_images=mosaic_max_cached_images,  # note
        random_pop=False,  # note
        pad_val=114.0,
    ),
    dict(type=Resize, keep_ratio=True, scale=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(
        type=Bbox2FomoMask, downsample_factor=downsample_factor, num_classes=num_classes
    ),
    dict(
        type=PackDetInputs,
        meta_keys=(
            "fomo_mask",
            "img_path",
            "img_id",
            "instances",
            "img_shape",
            "ori_shape",
            "gt_bboxes",
            "gt_bboxes_labels",
        ),
    ),
]

test_pipeline = [
    *pre_transform,
    dict(type=Resize, keep_ratio=True, scale=imgsz),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(
        type=Bbox2FomoMask, downsample_factor=downsample_factor, num_classes=num_classes
    ),
    dict(
        type=PackDetInputs,
        meta_keys=(
            "fomo_mask",
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=train_data),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=metainfo,
    ),
)
val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img=val_data),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline,
        metainfo=metainfo,
    ),
)
test_dataloader = val_dataloader

# data_preprocessor=dict(type='mmdet.DetDataPreprocessor')
# optimizer


find_unused_parameters = True

optim_wrapper = dict(
    optimizer=dict(
        # type=Adam, lr=lr, betas=momentum, weight_decay=weight_decay, eps=1e-7
        type=SGD,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    ),
)


# evaluator
val_evaluator = dict(type=FomoMetric)
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=epochs)


# learning policy
param_scheduler = [
    dict(type=LinearLR, begin=0, end=30, start_factor=0.001, by_epoch=False),  # warm-up
    dict(
        type=MultiStepLR,
        begin=1,
        end=100,
        milestones=[50, 70, 90],
        gamma=0.3,
        by_epoch=True,
    ),
]
# cfg=dict(compile=True)
