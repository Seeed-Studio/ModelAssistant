_base_ = '../_base_/default_runtime_cls.py'

default_scope = 'edgelab'

num_classes = 3
num_axes = 3
window_size = 30
stride = 20

model = dict(
    type='AccelerometerClassifier',
    backbone=dict(
        type='AxesNet',
        num_axes=num_axes,
        window_size=window_size,
        num_classes=num_classes,
    ),
    head=dict(
        type='edgelab.AxesClsHead',
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
)

# dataset settings
dataset_type = 'edgelab.SensorDataset'
data_root = './datasets/aixs-export'
batch_size = 1
workers = 1

shape = [1, num_axes * window_size]

train_pipeline = [
    # dict(type='edgelab.LoadSensorFromFile'),
    dict(type='edgelab.PackSensorInputs'),
]

test_pipeline = [
    # dict(type='edgelab.LoadSensorFromFile'),
    dict(type='edgelab.PackSensorInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='training',
        ann_file='info.labels',
        window_size=window_size,
        stride=stride,
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)


val_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        data_prefix='testing',
        ann_file='info.labels',
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_evaluator = dict(type='mmcls.Accuracy', topk=(1))


# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator


# optimizer
lr = 0.0005
epochs = 10

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=lr, betas=[0.9, 0.99], weight_decay=0))


train_cfg = dict(by_epoch=True, max_epochs=epochs)

val_cfg = dict()
test_cfg = dict()

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='edgelab.SensorClsVisualizer', vis_backends=vis_backends, name='visualizer')
