_base_ = './base.py'

# ========================Suggested optional parameters========================
# MODEL
gray = False
num_classes = 3

# ================================END=================================


model = dict(
    type='sscma.ImageClassifier',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0.0] if gray else [0.0, 0.0, 0.0],
        std=[255.0] if gray else [255.0, 255.0, 255.0],
    ),
    backbone=dict(type='MicroNet', rep=False, arch='l', gray=gray),
    head=dict(
        type='mmcls.LinearClsHead',
        in_channels=248,
        num_classes=num_classes,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5) if num_classes > 5 else 1,
    ),
)
