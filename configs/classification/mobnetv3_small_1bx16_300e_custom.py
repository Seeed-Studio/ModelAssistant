_base_ = "../_base_/default_runtime_cls.py"
default_scope = "edgelab"
custom_imports = dict(imports=["edgelab"], allow_failed_imports=False)

num_classes = 10

model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='mmcls.MobileNetV3', arch='small'),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.StackedLinearClsHead',
        num_classes=num_classes,
        in_channels=576,
        mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type='mmcls.HSwish'),
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(
            type='mmcls.Normal', layer='Linear', mean=0., std=0.01, bias=0.),
        topk=(1, 5))
)

# dataset settings
dataset_type = "mmcls.CustomDataset"
data_root = ""
height = 96
width = 96
batch_size = 16
workers = 1

data_root='/home/hongtai/open-mmlab/EdgeLab/datasets/fruit/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type="mmdet.Resize", scale=(height, width)),
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type="mmdet.Resize", scale=(height, width)),
    dict(type='mmcls.PackClsInputs'),
]

train_dataloader = dict(
    # Training dataset configurations
     batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train/',
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
     batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='valid/',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)


test_dataloader = val_dataloader

# optimizer
lr = 0.001
epochs = 300

find_unused_parameters = True

optim_wrapper = dict(
    optimizer=dict(type="Adam", lr=lr, weight_decay=5e-4, eps=1e-7))

# evaluator
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, 5))
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=epochs)

val_cfg = dict()
test_cfg = dict()

# learning policy
param_scheduler = [
    dict(type="LinearLR", begin=0, end=30, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(
        type="MultiStepLR",
        begin=1,
        end=500,
        milestones=[100, 200, 250],
        gamma=0.1,
        by_epoch=True,
    ),
]
