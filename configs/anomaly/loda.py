_base_ = '../_base_/default_runtime_cls.py'


window_size = 30
stride = 10
num_axes = 3

model = dict(type='edgelab.LODA', num_bins=10, num_cuts=100, yield_rate=0.9)


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
        pack=True,
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)


val_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    shuffle=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        window_size=window_size,
        stride=stride,
        pack=False,
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

lr = 0.1
epochs = 1
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=lr, betas=[0.9, 0.99], weight_decay=0))

train_cfg = dict(by_epoch=True, max_epochs=epochs)

val_cfg = dict()
test_cfg = dict()
