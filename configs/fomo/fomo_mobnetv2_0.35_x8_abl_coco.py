_base_ = '../_base_/pose_default_runtime.py'

custom_imports = dict(imports=['edgelab'],
                      allow_failed_imports=False)
num_classes=2
model = dict(
    type='Fomo',
    backbone=dict(type='MobileNetV2', widen_factor=0.35, out_indices=(2, )),
    head=dict(
        type='Fomo_Head',
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
data_root = '132'
height=96
width=96
batch_size=32
workers=4


train_pipeline = [
    dict(type='RandomResizedCrop', height=height, width=width, scale=(0.90, 1.1),
         p=1),
    dict(type='Rotate', limit=20),
    dict(type='RandomBrightnessContrast',
         brightness_limit=0.2,
         contrast_limit=0.2,
         p=0.5),
    dict(type='HorizontalFlip', p=0.5),
]
test_pipeline = [dict(type='Resize', height=height, width=width, p=1)]

data = dict(samples_per_gpu=batch_size,
            workers_per_gpu=workers,
            train_dataloader=dict(collate=True),
            val_dataloader=dict(collate=True),
            train=dict(type=dataset_type,
                       data_root=data_root,
                       ann_file='train/_annotations.coco.json',
                       img_prefix='train',
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     data_root=data_root,
                     test_mode=True,
                     ann_file='valid/_annotations.coco.json',
                     img_prefix='valid',
                     pipeline=test_pipeline),
            test=dict(type=dataset_type,
                      data_root=data_root,
                      test_mode=True,
                      ann_file='valid/_annotations.coco.json',
                      img_prefix='valid',
                      pipeline=test_pipeline))

# optimizer
lr=0.001
epochs=300
optimizer = dict(type='Adam', lr=lr, weight_decay=0.0005)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=30,
                 warmup_ratio=0.000001,
                 step=[100, 200, 250])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
evaluation = dict(interval=1, metric=['mAP'], fomo=True)
find_unused_parameters = True

log_config = dict(interval=5,
                  hooks=[dict(type='TensorboardLoggerHook', ndigits=4)])