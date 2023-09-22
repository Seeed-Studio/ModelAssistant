_base_ = ['../_base_/default_runtime_det.py']
default_scope = 'mmdet'

# ========================Suggested optional parameters========================

num_classes = 20

# dataset settings
dataset_type = 'sscma.CustomVocdataset'
data_root = 'https://universe.roboflow.com/ds/hbQfAl4XVj?key=jepUkAFOqo'

train_ann = 'train/'
train_data = 'train/'
val_ann = 'valid/'
val_data = 'valid/'

height = 352
width = 352
batch = 32
workers = 4

val_batch = 1
val_workers = 1
persistent_workers = True

lr = 0.001
epochs = 300
weight_decay = 0.0005
momentum = (0.9, 0.99)

# ================================END=================================

model = dict(
    type='FastestDet',
    backbone=dict(
        type='CusShuffleNetV2',
        out_indices=(0, 1, 2),
        widen_factor=0.25,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    ),
    neck=dict(type='sscma.SPP', input_channels=336, output_channels=96, layers=[1, 2, 3]),
    bbox_head=dict(
        type='sscma.Fastest_Head',
        input_channels=96,
        num_classes=num_classes,
    ),
    # training and testing settings
    train_cfg=dict(assigner=dict(type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100,
    ),
)


backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=[0.0, 0.0, 0.0], to_rgb=True, ratio_range=(1, 4)),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    dict(type='Resize', scale=(height, width), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(height, width), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=train_data),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img=val_data),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=lr, betas=momentum, weight_decay=weight_decay, eps=1e-7),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', warmup='linear', warmup_iters=8000, warmup_ratio=0.000001, step=[100, 200, 250])
# runtime settings
evaluation = dict(interval=1, metric=['mAP'])
find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (24 samples per GPU)
auto_scale_lr = dict(base_batch_size=192)

val_evaluator = dict(type='FomoMetric')
test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
