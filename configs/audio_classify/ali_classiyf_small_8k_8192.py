_base_ = ['../_base_/default_runtime.py']

words = [
    "no",
    "off",
    "on",
    "yes",
]
words = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin',
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
    'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]
# model settings
num_classes=35
model = dict(type='Audio_classify',
             backbone=dict(type='SoundNetRaw',
                           nf=2,
                           clip_length=64,
                           factors=[4, 4, 4],
                           out_channel=36),
             head=dict(type='Audio_head',
                       in_channels=36,
                       n_classes=num_classes,
                       drop=0.2),
             loss_cls=dict(type='LabelSmoothCrossEntropyLoss',
                           reduction='sum',
                           smoothing=0.1))

# dataset settings
dataset_type = 'Speechcommand'

transforms = [
    'amp', 'neg', 'tshift', 'tmask', 'ampsegment', 'cycshift', 'awgn', 'abgn',
    'apgn', 'argn', 'avgn', 'aun', 'phn', 'sine'
]

data_root = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
width=8192
batch_size=128
workers=8

train_pipeline = dict(type='AudioAugs', k_augs=transforms)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop',
         crop_size=224,
         efficientnet_style=True,
         interpolation='bicubic',
         backend='pillow'),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers,
    train=dict(type=dataset_type,
               root=data_root,
               sampling_rate=8000,
               segment_length=width,
               pipeline=train_pipeline,
               mode='train',
               use_background=True,
               lower_volume=True,
               words=words),
    val=dict(type=dataset_type,
             root=data_root,
             sampling_rate=8000,
             segment_length=width,
             mode='val',
             use_background=False,
             lower_volume=True,
             words=words),
    test=dict(
        type=dataset_type,
        root=data_root,
        sampling_rate=8000,
        segment_length=width,
        mode='test',
        use_background=False,
        lower_volume=True,
        words=words))

custom_hooks = dict(type='Audio_hooks',
                    n_cls=len(words),
                    multilabel=False,
                    loss=dict(type='LabelSmoothCrossEntropyLoss',
                              reduction='sum',
                              smoothing=0.1),
                    seq_len=width,
                    sampling_rate=8000,
                    device='0',
                    augs_mix=['mixup', 'timemix', 'freqmix', 'phmix'],
                    mix_ratio=1,
                    local_rank=0,
                    epoch_mix=12,
                    mix_loss='bce',
                    priority=0)

evaluation = dict(save_best='acc',
                  interval=1,
                  metric='accuracy',
                  metric_options={'topk': (1, )})

# optimizer
lr=0.0003
epochs=1500
optimizer = dict(type='AdamW',
                 lr=lr,
                 betas=[0.9, 0.99],
                 weight_decay=0,
                 eps=1e-8)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(policy='OneCycleLR', max_lr=3e-4, steps_per_epoch=200, epochs=1000, pct_start=0.1, )
# lr_config = dict(policy='step', step=[50,200])
lr_config = dict(
    policy='OneCycle',
    max_lr=lr,
    # steps_per_epoch=388,
    # epoch=1500,
    pct_start=0.1)
