# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *
    from .._base_.schedules.schedule_1x import *

from sscma.datasets import MeterData
from sscma.models import PFLD, PfldMobileNetV2, PFLDhead, PFLDLoss
from torch.nn import ReLU
from albumentations import (
    Resize,
    ColorJitter,
    MedianBlur,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    Affine,
)
from mmengine.optim import LinearLR, MultiStepLR
from torch.optim import Adam
from sscma.evaluation import PointMetric
from mmengine.dataset import default_collate
from sscma.deploy.models.pfld_infer import PFLDInfer

# ========================Suggested optional parameters========================
# MODEL
num_classes = 1
deepen_factor = 0.33
widen_factor = 0.15

# DATA
dataset_type = MeterData

data_root = ""

train_ann = "train/annotations.txt"
train_data = "train/images"
val_ann = "val/annotations.txt"
val_data = "val/images"

height = 112
width = 112
imgsz = (width, height)

# TRAIN
batch = 32
workers = 4
val_batch = 32
val_workers = 2
lr = 0.0001
epochs = 1000
weight_decay = 1e-6
momentum = (0.9, 0.99)

persistent_workers = True
# ================================END=================================

model = dict(
    type=PFLD,
    backbone=dict(
        type=PfldMobileNetV2,
        inchannel=3,
        layer1=[16, 16, 16, 16, 16],
        layer2=[32, 32, 32, 32, 32, 32],
        out_channel=32,
    ),
    head=dict(
        type=PFLDhead,
        num_point=num_classes,
        input_channel=32,
        act_cfg=ReLU,
        loss_cfg=dict(type=PFLDLoss),
    ),
)


deploy = dict(
    type=PFLDInfer,
)

train_pipeline = [
    dict(type=Resize, height=imgsz[1], width=imgsz[0], interpolation=0),
    # dict(type="PixelDropout"),
    dict(type=ColorJitter, brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
    # dict(type="CoarseDropout",max_height=12,max_width=12),
    # dict(type='GaussNoise'),
    dict(type=MedianBlur, blur_limit=3, p=0.5),
    dict(type=HorizontalFlip),
    dict(type=VerticalFlip),
    dict(type=Rotate, limit=45, p=0.7),
    dict(type=Affine, translate_percent=[0.05, 0.30], p=0.6),
]

val_pipeline = [dict(type=Resize, height=imgsz[1], width=imgsz[0])]


train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    collate_fn=default_collate,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_data,
        index_file=train_ann,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    collate_fn=default_collate,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=val_data,
        index_file=val_ann,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader


evaluation = dict(save_best="loss")
optim_wrapper = dict(
    optimizer=dict(type=Adam, lr=lr, betas=momentum, weight_decay=weight_decay)
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
val_evaluator = dict(type=PointMetric)
test_evaluator = val_evaluator

find_unused_parameters = True

train_cfg = dict(by_epoch=True, max_epochs=epochs)


# learning policy
param_scheduler = [
    dict(
        type=LinearLR, begin=0, end=500, start_factor=0.001, by_epoch=False
    ),  # warm-up
    dict(
        type=MultiStepLR,
        begin=1,
        end=epochs,
        milestones=[350, 500, 600, 700, 850, 1050,1300,1400],
        gamma=0.1,
        by_epoch=True,
    ),
]
