_base_ = '../_base_/pose_default_runtime.py'
custom_imports = dict(imports=['models', 'datasets'], allow_failed_imports=False)
model = dict(
    type='PFLD',
    backbone=dict(
        type='PFLDInference',
    ),
    loss_cfg=dict(type='PFLDLoss')
)

# dataset settings
dataset_type = 'MeterData'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        index_file='datasets/table/train_data/list_d.txt',
        transform=True
    ),
    val=dict(
        type=dataset_type,
        index_file=r'datasets/table/test_data/list_d.txt',
        transform=False
    ),
    test=dict(
        type=dataset_type,
        index_file=r'datasets/table/test_data/list_d.txt',
        transform=False
    )
)

evaluation = dict(save_best='loss')
optimizer = dict(type='Adam', lr=0.0001, betas=(0.9, 0.99), weight_decay=1e-6)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.0001,
    step=[440, 490])
# runtime settings
# runner = dict(type='EpochBasedRunner', max_epochs=30)
# evaluation = dict(interval=1, metric=['bbox'])
total_epochs = 500
find_unused_parameters = True
