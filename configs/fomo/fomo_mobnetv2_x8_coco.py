_base_ = '../_base_/pose_default_runtime.py'

custom_imports = dict(imports=['models', 'datasets', 'core'],
                      allow_failed_imports=False)

model = dict(
    type='Fomo',
    backbone=dict(type='MobileNetV2', widen_factor=0.35, out_indices=(2, )),
    head=dict(type='Fomo_Head',
              input_channels=16,
              num_classes=80,
              middle_channels=[96, 32],
              act_cfg='ReLU6',
              loss_cls=dict(type='BCEWithLogitsLoss', reduction='mean'),
              loss_bg=dict(type='BCEWithLogitsLoss', reduction='mean'),
              cls_weight=100),
)

# dataset settings
dataset_type = 'CustomCocoDataset'
data_root = (
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip")

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=(96, 96),
         multiscale_mode='range',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(96, 96),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            #  dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',  # use RepeatDataset to speed up training
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            filter_empty_gt=False,
            ann_file='annotations/instances_train2017.json',
            img_prefix='train2017/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        filter_empty_gt=False,
        ann_file='annotations/instances_val2017.json',
        img_prefix='test2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        filter_empty_gt=False,
        ann_file='annotations/instances_val2017.json',
        img_prefix='test2017/',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='Adam', lr=0.003, weight_decay=0.0005)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=5000,
                 warmup_ratio=0.0000001,
                 step=[100, 200, 250])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)
evaluation = dict(interval=1, metric=['mAP'], fomo=True)
find_unused_parameters = True

log_config = dict(interval=2,
                  hooks=[dict(type='TensorboardLoggerHook', ndigits=4)])