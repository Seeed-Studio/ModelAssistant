_base_ = ['../_base_/cls_default_runtime.py']

# model settings
custom_imports = dict(imports=['models', 'datasets','core'], allow_failed_imports=False)

model = dict(
    type='Audio_classify',
    backbone=dict(type='SoundNetRaw', nf=2, clip_length=64, factors=[4, 4, 4], out_channel=36),
    head=dict(type='Audio_head', in_channels=36, n_classes=4, drop=0.2),
    loss_cls=dict(type='LabelSmoothCrossEntropyLoss', reduction='sum', smoothing=0.1)
)

# dataset settings
dataset_type = 'Speechcommand'

transforms = ['amp', 'neg', 'tshift', 'tmask', 'ampsegment', 'cycshift', 'awgn', 'abgn', 'apgn', 'argn', 'avgn', 'aun',
              'phn', 'sine']

data_root = '/home/dq/github/datasets/speech_commands_v0.02'
train_pipeline = dict(type='AudioAugs', k_augs=transforms)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=224,
        efficientnet_style=True,
        interpolation='bicubic',
        backend='pillow'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

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
        # pipeline=test_pipeline,
        use_background=False))


custom_hooks = dict(type='Audio_hooks',n_cls=4,multilabel=False,loss=dict(type='LabelSmoothCrossEntropyLoss', reduction='sum', smoothing=0.1),
        seq_len=8192,sampling_rate=8000,device='0',augs_mix=['mixup', 'timemix', 'freqmix', 'phmix'],mix_ratio=1,
        local_rank=0,epoch_mix=12,mix_loss='bce',priority=0)


evaluation = dict(save_best='acc',
                  interval=1, metric='accuracy', metric_options={'topk': (1,)})

# optimizer
optimizer = dict(type='AdamW', lr=0.0003, betas=[0.9, 0.99], weight_decay=0, eps=1e-8)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='OneCycleLR', max_lr=3e-4, steps_per_epoch=200, epochs=1000, pct_start=0.1, )
# lr_config = dict(policy='step', step=[50,200])
lr_config = dict(policy='OneCycle',
                    max_lr=0.0003,
                    # steps_per_epoch=388,
                    # epoch=1500,
                    pct_start=0.1)

