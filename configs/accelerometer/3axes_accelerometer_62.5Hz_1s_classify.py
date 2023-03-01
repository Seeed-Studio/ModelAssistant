_base_ = ['../_base_/default_runtime.py']

num_classes = 3
model = dict(type='AccelerometerClassifier',
             backbone=dict(type='AxesNet',
                           num_axes=3,
                           frequency=62.5,
                           duration=1,
                           out_channels=256,
                           ),
             head=dict(type='LinearClsHead',
                       in_channels=256,
                       num_classes=num_classes,
                       loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                       topk=(1, 5),
                       ))

# dataset settings
dataset_type = 'AxesDataset'
data_root = './work_dirs/datasets/axes'
batch_size = 4
workers = 2


data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers,
    train=dict(type=dataset_type,
               data_root=data_root,
               label_file='training/info.labels',
               mode='train',
               ),
    val=dict(type=dataset_type,
             data_root=data_root,
             label_file='testing/info.labels',
             mode='val',
             ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        label_file='testing/info.labels',
        mode='test',
    ))


evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': 1})

# optimizer
lr = 0.0005
epochs = 100
optimizer = dict(type='Adam',
                 lr=lr,
                 betas=[0.9, 0.99],
                 weight_decay=0,
                 )

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='OneCycle',
    max_lr=lr,
    pct_start=0.1)
