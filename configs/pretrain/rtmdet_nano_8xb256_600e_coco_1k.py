# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .default_runtime import *
    from .imagenet_bs2048_rsb import *

from torch.nn import SyncBatchNorm
from torch.nn.modules.activation import SiLU

from sscma.models.backbones.cspnext import CSPNeXt

from mmpretrain.models.classifiers import ImageClassifier
from mmpretrain.datasets import ImageNet
from mmengine.dataset.sampler import DefaultSampler
from mmpretrain.datasets.transforms import (
    LoadImageFromFile,
    RandomResizedCrop,
    RandomFlip,
    RandAugment,
    PackInputs,
    ResizeEdge,
    CenterCrop,
)
from mmpretrain.evaluation.metrics import Accuracy
from mmpretrain.models import GlobalAveragePooling, LinearClsHead, LabelSmoothLoss
from mmpretrain.models import LabelSmoothLoss, Mixup, CutMix
from mmpretrain.models import ClsDataPreprocessor


d_factor = 0.33
w_factor = 0.25
input_shape = 320
checkpoint = "/home/dq/code/sscma/pretrain.pt"  # noqa

model = dict(
    type=ImageClassifier,
    data_preprocessor=dict(
        type=ClsDataPreprocessor,
        num_classes=1000,
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        to_rgb=True,
    ),
    backbone=dict(
        type=CSPNeXt,
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=d_factor,
        widen_factor=w_factor,
        channel_attention=True,
        use_depthwise=True,
        norm_cfg=dict(type=SyncBatchNorm),
        act_cfg=dict(type=SiLU, inplace=True),
        init_cfg=dict(type="Pretrained", prefix="backbone.", checkpoint=checkpoint),
        
    ),
    neck=dict(type=GlobalAveragePooling),
    head=dict(
        type=LinearClsHead,
        num_classes=1000,
        in_channels=256,
        loss=dict(
            type=LabelSmoothLoss,
            label_smooth_val=0.1,
            mode="original",
            loss_weight=1.0,
        ),
        topk=(1, 5),
    ),
    train_cfg=dict(
        augments=[
            dict(type=Mixup, alpha=0.2),
            dict(type=CutMix, alpha=1.0),
        ]
    ),
)

# dataset settings
data_root = "/home/dq/datasets/imagenet/"
dataset_type = "ImageNet"
data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor["mean"][::-1]
bgr_std = data_preprocessor["std"][::-1]

train_pipeline = [
    dict(type=LoadImageFromFile),
    dict(type=RandomResizedCrop, scale=320, backend="pillow", interpolation="bicubic"),
    dict(type=RandomFlip, prob=0.5, direction="horizontal"),
    dict(
        type=RandAugment,
        policies="timm_increasing",
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean], interpolation="bicubic"),
    ),
    dict(type=PackInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile),
    dict(
        type=ResizeEdge,
        scale=236,
        edge="short",
        backend="pillow",
        interpolation="bicubic",
    ),
    dict(type=CenterCrop, crop_size=320),
    dict(type=PackInputs),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=12,
    dataset=dict(
        type=ImageNet, data_root=data_root, split="train", pipeline=train_pipeline
    ),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=12,
    dataset=dict(
        type=ImageNet, data_root=data_root, split="val", pipeline=test_pipeline
    ),
    sampler=dict(type=DefaultSampler, shuffle=False),
)
val_evaluator = dict(type=Accuracy, topk=(1, 5))
# resume = "/home/dq/code/sscma/work_dirs/rtmdet_nano_8xb256_600e_coco_1k/epoch_310.pth"
# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
