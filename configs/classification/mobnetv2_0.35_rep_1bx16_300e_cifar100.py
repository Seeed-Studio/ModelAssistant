_base_ = '../_base_/default_runtime_cls.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='MobileNetv2', widen_factor=0.35, out_indices=(2,), rep=True),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        in_channels=16,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
    ),
)
