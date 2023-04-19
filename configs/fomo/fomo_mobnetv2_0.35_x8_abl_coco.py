_base_ = '../_base_/default_runtime_det.py'
default_scope = 'edgelab'
custom_imports = dict(imports=['edgelab'], allow_failed_imports=False)

num_classes = 2
model = dict(
    type='Fomo',
    backbone=dict(type='mmdet.MobileNetV2', widen_factor=0.35, out_indices=(2, )),
    head=dict(
        type='FomoHead',
        input_channels=16,
        num_classes=num_classes,
        middle_channels=[96, 32],
        act_cfg='ReLU6',
        loss_cls=dict(type='BCEWithLogitsLoss',
                      reduction='none',
                      pos_weight=40),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
        cls_weight=40,
    ),
)

# dataset settings
dataset_type = 'FomoDatasets'
data_root = '/home/dq/datasets/mask/'
height = 96
width = 96
batch_size = 16
workers = 1

train_pipeline = [
    dict(type='RandomResizedCrop',
         height=height,
         width=width,
         scale=(0.90, 1.1),
         p=1),
    dict(type='Rotate', limit=20),
    dict(type='RandomBrightnessContrast',
         brightness_limit=0.2,
         contrast_limit=0.2,
         p=0.5),
    dict(type='HorizontalFlip', p=0.5),
]
test_pipeline = [dict(type='Resize', height=height, width=width, p=1)]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='fomo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
               data_root=data_root,
               ann_file='train/_annotations.coco.json',
               img_prefix='train',
               pipeline=train_pipeline),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='fomo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
                               data_root=data_root,
                               test_mode=True,
                               ann_file='valid/_annotations.coco.json',
                               img_prefix='valid',
                               pipeline=test_pipeline), )
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
