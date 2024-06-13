# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = '../_base_/default_runtime_pose.py'

# ========================Suggested optional parameters========================
# MODEL
num_classes = 4
deepen_factor = 0.33
widen_factor = 0.15

# DATA
dataset_type = 'MeterData'
data_root = ''
train_ann = 'train/annotations.txt'
train_data = 'train/images'
val_ann = 'val/annotations.txt'
val_data = 'val/images'

height = 192
width = 192
imgsz = (width, height)

# TRAIN
batch = 32
workers = 4
val_batch = 1
val_workers = 1

lr = 0.0001
epochs = 1000
weight_decay = 1e-6
momentum = (0.9, 0.99)

persistent_workers = True
# ================================END=================================


norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)

model = dict(
    type='PFLD',
    backbone=dict(
        type='sscma.YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='sscma.YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    head=dict(type='PFLDhead', num_point=num_classes, input_channel=40, act_cfg='ReLU', loss_cfg=dict(type='PFLDLoss')),
)


train_pipeline = [
    dict(type='Resize', height=imgsz[1], width=imgsz[0], interpolation=0),
    # dict(type="PixelDropout"),
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
    # dict(type="CoarseDropout",max_height=12,max_width=12),
    # dict(type='GaussNoise'),
    dict(type='MedianBlur', blur_limit=3, p=0.5),
    dict(type='HorizontalFlip'),
    dict(type='VerticalFlip'),
    dict(type='Rotate', limit=45, p=0.7),
    dict(type='Affine', translate_percent=[0.05, 0.30], p=0.6),
]

val_pipeline = [dict(type='Resize', height=imgsz[1], width=imgsz[0])]

train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
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
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=val_data,
        index_file=val_ann,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader


evaluation = dict(save_best='loss')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=lr, betas=momentum, weight_decay=weight_decay))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
val_evaluator = dict(type='PointMetric')
test_evaluator = val_evaluator

find_unused_parameters = True

train_cfg = dict(by_epoch=True, max_epochs=epochs)
# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR', begin=1, end=500, milestones=[300, 400, 500, 600, 700, 800, 900], gamma=0.1, by_epoch=True
    ),
]
