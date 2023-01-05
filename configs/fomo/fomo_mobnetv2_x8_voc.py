_base_ = '../_base_/pose_default_runtime.py'

custom_imports = dict(imports=['models', 'datasets', 'core'],
                      allow_failed_imports=False)

model = dict(
    type='Fomo',
    backbone=dict(type='MobileNetV2', widen_factor=0.35, out_indices=(2, )),
    head=dict(type='Fomo_Head',
              input_channels=16,
              num_classes=20,
              middle_channels=[96, 32],
              act_cfg='ReLU6',
              loss_cls=dict(type='BCEWithLogitsLoss', reduction='mean'),
              loss_bg=dict(type='BCEWithLogitsLoss', reduction='mean'),
              cls_weight=100),
)

# dataset settings
dataset_type = 'CustomVocdataset'
data_root = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand',
         mean=img_norm_cfg['mean'],
         to_rgb=img_norm_cfg['to_rgb'],
         ratio_range=(1, 2)),
    dict(type='MinIoURandomCrop',
         min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
         min_crop_size=0.3),
    dict(type='Resize',
         img_scale=[(96, 96)],
         multiscale_mode='range',
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(96, 96),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='PhotoMetricDistortion'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='DefaultFormatBundle'),
             dict(type='Collect', keys=['img'])
         ])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',  # use RepeatDataset to speed up training
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='ImageSets/Main/train.txt',
            #  img_prefix=None,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/val.txt',
        #  img_prefix=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/val.txt',
        #   img_prefix=None,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0005)

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


log_config = dict(interval=5,
                  hooks=[dict(type='TensorboardLoggerHook', ndigits=4)])