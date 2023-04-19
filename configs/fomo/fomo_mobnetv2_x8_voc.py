_base_ = '../_base_/default_runtime_det.py'

custom_imports = dict(imports=['models', 'datasets', 'core'],
                      allow_failed_imports=False)

model = dict(
    type='Fomo',
    backbone=dict(type='MobileNetV2', widen_factor=0.35, out_indices=(2, )),
    head=dict(type='FomoHead',
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

img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
         img_scale=[(96, 96)],
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
    dict(type='MultiScaleFlipAug',
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
batch_size = 16
workers = 2
data = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='fomo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 filter_empty_gt=False,
                 ann_file='annotations/instances_train2017.json',
                 img_prefix='train2017/',
                 pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='fomo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 filter_empty_gt=False,
                 ann_file='annotations/instances_val2017.json',
                 img_prefix='test2017/',
                 pipeline=test_pipeline),
)
test_dataloader = val_dataloader

# optimizer
lr = 0.001
epochs = 300

find_unused_parameters = True

optim_wrapper=dict(optimizer = dict(type='Adam', lr=lr, weight_decay=5e-4,eps=1e-7))

#evaluator
val_evaluator=dict(
    type='FomoMetric')
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True,max_epochs=70)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=30, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=1,
        end=500,
        milestones=[100, 200,250],
        gamma=0.1,
        by_epoch=True)
]
