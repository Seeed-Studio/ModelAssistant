# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = '../_base_/default_runtime_cls.py'

# ========================Suggested optional parameters========================
# MODEL
window_size = 30
stride = 10
num_axes = 3
model = dict(type='sscma.LODA', num_bins=10, num_cuts=100, yield_rate=0.9)

# DATA
dataset_type = 'sscma.SensorDataset'
data_root = 'datasets/aixs-export'

train_ann = 'info.labels'
train_data = 'training'
val_ann = 'info.labels'
val_data = 'testing'

# TRAIN
batch = 1
workers = 1
val_batch = 1
val_workers = 1

lr = 0.1
epochs = 1

weight_decay = 0.0005
momentum = (0.9, 0.99)
# ================================END=================================

shape = [1, num_axes * window_size]

train_pipeline = [
    # dict(type='sscma.LoadSensorFromFile'),
    dict(type='sscma.PackSensorInputs'),
]

test_pipeline = [
    # dict(type='sscma.LoadSensorFromFile'),
    dict(type='sscma.PackSensorInputs'),
]

train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=train_data,
        ann_file=train_ann,
        window_size=window_size,
        stride=stride,
        pack=True,
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)


val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    shuffle=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        pack=False,
        data_prefix=val_data,
        ann_file=val_ann,
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_evaluator = dict(type='mmcls.Accuracy', topk=(1,))


# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=lr, betas=momentum, weight_decay=weight_decay),
)

train_cfg = dict(by_epoch=True, max_epochs=epochs)

val_cfg = dict()
test_cfg = dict()
