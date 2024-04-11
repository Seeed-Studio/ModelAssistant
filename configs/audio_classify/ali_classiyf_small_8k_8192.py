# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = ['../_base_/default_runtime_cls.py']

words = [
    'no',
    'off',
    'on',
    'yes',
]
words = [
    'backward',
    'bed',
    'bird',
    'cat',
    'dog',
    'down',
    'eight',
    'five',
    'follow',
    'forward',
    'four',
    'go',
    'happy',
    'house',
    'learn',
    'left',
    'marvin',
    'nine',
    'no',
    'off',
    'on',
    'one',
    'right',
    'seven',
    'sheila',
    'six',
    'stop',
    'three',
    'tree',
    'two',
    'up',
    'visual',
    'wow',
    'yes',
    'zero',
]
# ========================Suggested optional parameters========================
# MODEL
num_classes = 35

# DATA
dataset_type = 'Speechcommand'
data_root = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
width = 8192

# TRAIN
batch = 128
workers = 6
persistent_workers = True

val_batch = batch
val_workers = workers

lr = 0.0003
epochs = 1500

weight_decay = 0.0005
momentum = (0.9, 0.99)
# ================================END=================================


model = dict(
    type='Audio_classify',
    n_cls=num_classes,
    multilabel=False,
    loss=dict(type='LabelSmoothCrossEntropyLoss', reduction='sum', smoothing=0.1),
    backbone=dict(type='SoundNetRaw', nf=2, clip_length=64, factors=[4, 4, 4], out_channel=36),
    head=dict(type='Audio_head', in_channels=36, n_classes=num_classes, drop=0.2),
    loss_cls=dict(type='LabelSmoothCrossEntropyLoss', reduction='sum', smoothing=0.1),
)


transforms = [
    'amp',
    'neg',
    'tshift',
    'tmask',
    'ampsegment',
    'cycshift',
    'awgn',
    'abgn',
    'apgn',
    'argn',
    'avgn',
    'aun',
    'phn',
    'sine',
]


train_pipeline = dict(type='AudioAugs', k_augs=transforms)


train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        root=data_root,
        sampling_rate=8000,
        segment_length=width,
        pipeline=train_pipeline,
        mode='train',
        use_background=True,
        lower_volume=True,
        words=words,
    ),
)
val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        root=data_root,
        sampling_rate=8000,
        segment_length=width,
        mode='val',
        use_background=False,
        lower_volume=True,
        words=words,
    ),
)
test_dataloader = val_dataloader


data_preprocessor = dict(
    type='ETADataPreprocessor',
    n_cls=len(words),
    multilabel=False,
    seq_len=width,
    sampling_rate=8000,
    augs_mix=['mixup', 'timemix', 'freqmix', 'phmix'],
    mix_ratio=1,
    local_rank=0,
    epoch_mix=12,
    mix_loss='bce',
)

# optimizer

find_unused_parameters = True

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=lr, betas=momentum, weight_decay=weight_decay, eps=1e-7),
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# evaluator
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, 5) if num_classes > 5 else 1)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000)
val_cfg = dict()
test_cfg = dict()
# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=30, start_factor=0.001, by_epoch=False),  # warm-up
    dict(type='MultiStepLR', begin=1, end=500, milestones=[100, 200, 250], gamma=0.1, by_epoch=True),
]
