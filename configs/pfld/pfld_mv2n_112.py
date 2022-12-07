_base_ = '../_base_/pose_default_runtime.py'
custom_imports = dict(imports=['models', 'datasets','core'],
                      allow_failed_imports=False)
model = dict(type='PFLD',
             backbone=dict(type='PFLDInference', ),
             loss_cfg=dict(type='PFLDLoss'))

train_pipeline = [
    dict(type="Resize", height=112, width=112,interpolation=0),
    dict(type='ColorJitter', brightness=0.3, p=0.5),
    # dict(type='GaussNoise'),
    dict(type='MedianBlur', blur_limit=3, p=0.3),
    dict(type='HorizontalFlip'),
    dict(type='VerticalFlip'),
    dict(type='Rotate'),
    dict(type='Affine', translate_percent=[0.05, 0.1], p=0.6)
]

val_pipeline=[
    dict(type="Resize", height=112, width=112)
]

# dataset settings
dataset_type = 'MeterData'

data_root='~/datasets/meter'

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        index_file=r'train/annotations.txt',
        pipeline=train_pipeline,
        test_mode=False),
    val=dict(type=dataset_type,
             data_root=data_root,
             index_file=r'val/annotations.txt',
             pipeline=val_pipeline,
             test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        index_file=r'val/annotations.txt',
        pipeline=val_pipeline,
        test_mode=True
        # dataset_info={{_base_.dataset_info}}
    ))

evaluation = dict(save_best='loss')
optimizer = dict(type='Adam', lr=0.0001, betas=(0.9, 0.99), weight_decay=1e-6)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=400,
                 warmup_ratio=0.0001,
                 step=[150, 300, 450])
# lr_config = dict(
#     policy='OneCycle',
#     max_lr=0.0001,
#     # steps_per_epoch=388,
#     # epoch=1500,
#     pct_start=0.1)
# runtime settings
# runner = dict(type='EpochBasedRunner', max_epochs=30)
# evaluation = dict(interval=1, metric=['bbox'])
total_epochs = 500
find_unused_parameters = True
