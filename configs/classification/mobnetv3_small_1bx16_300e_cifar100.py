_base_ = './mobnetv2_1.0_1bx16_300e_cifar100.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='mmcls.MobileNetV3', arch='small', _delete_=True),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.StackedLinearClsHead',
        in_channels=576,
        mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type='mmcls.HSwish'),
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(type='mmcls.Normal', layer='Linear', mean=0.0, std=0.01, bias=0.0),
    ),
)
