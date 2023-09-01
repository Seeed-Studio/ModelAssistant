_base_ = './mobnetv2_1.0_1bx16_300e_custom.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# MODEL
gray = False

# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(
        type='EfficientNet', arch='b0', input_channels=1 if gray else 3, out_indices=(6,), rep=True, _delete_=True
    ),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        in_channels=320,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
    ),
)
