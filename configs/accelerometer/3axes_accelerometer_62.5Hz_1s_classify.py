# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './base.py'

default_scope = 'sscma'
# ========================Suggested optional parameters========================
# MODEL
num_classes = 3
num_axes = 3
window_size = 62
stride = 20

# DATA
dataset_type = 'sscma.SensorDataset'
data_root = 'datasets/sensor-export'
train_ann = 'info.labels'
train_data = 'training'
val_ann = 'info.labels'
val_data = 'testing'
batch = 1
workers = 1
val_batch = batch
val_workers = workers

# TRAIN
lr = 0.0005
epochs = 10
weight_decay = 0.0005
momentum = (0.9, 0.99)
# ================================END=================================


model = dict(
    type='AccelerometerClassifier',
    backbone=dict(
        type='AxesNet',
        num_axes=num_axes,
        window_size=window_size,
        num_classes=num_classes,
    ),
    head=dict(
        type='sscma.AxesClsHead',
        loss=dict(type='sscma.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5) if num_classes > 5 else 1,
    ),
)


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
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)


val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        data_prefix=val_data,
        ann_file=val_ann,
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_evaluator = dict(type='sscma.Accuracy', topk=(1, 5) if num_classes > 5 else 1)


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

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='sscma.SensorClsVisualizer', vis_backends=vis_backends, name='visualizer')
