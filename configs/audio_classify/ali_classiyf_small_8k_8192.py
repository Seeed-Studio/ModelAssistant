_base_ = ['../_base_/default_runtime.py']

# model settings
custom_imports = dict(imports=['models', 'datasets'], allow_failed_imports=False)

model = dict(
    type='Audio_classify',
    backbone=dict(type='SoundNetRaw', nf=2, clip_length=64, factors=[4, 4, 4], out_channel=48),
    head=dict(type='Audio_head', in_channels=48, n_classes=35, drop=0.2),
    loss_cls=dict(type='LabelSmoothCrossEntropyLoss', reduction='mean', smoothing=0.005)
)

# dataset settings
dataset_type = 'Speechcommand'

transforms = ['amp', 'neg', 'tshift', 'tmask', 'ampsegment', 'cycshift', 'awgn', 'abgn', 'apgn', 'argn', 'avgn', 'aun',
              'phn', 'sine']

data_root = 'D:\gitlab\datasets\yes\speech_commands_v0.02'
train_pipeline = dict(type='AudioAugs', k_augs=transforms)

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root=data_root,
        sampling_rate=8000,
        segment_length=8192,
        pipeline=train_pipeline,
        mode='train',
        use_background=True),
    val=dict(
        type=dataset_type,
        root=data_root,
        sampling_rate=8000,
        segment_length=8192,
        mode='val',
        use_background=False),
    test=dict(
        type=dataset_type,
        root=data_root,
        sampling_rate=8000,
        segment_length=8192,
        mode='test',
        use_background=False))

evaluation = dict(save_best='acc',
                  interval=1, metric='accuracy', metric_options={'topk': (1,)})

# optimizer
optimizer = dict(type='AdamW', lr=0.0003, betas=[0.9, 0.99], weight_decay=0, eps=1e-8)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='OneCycleLR', max_lr=3e-4, steps_per_epoch=200, epochs=1000, pct_start=0.1, )
lr_config = dict(policy='step', step=[15])

