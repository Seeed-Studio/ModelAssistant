_base_ = './base.py'

# ========================Suggested optional parameters========================
# MODEL
deepen_factor = 0.15
widen_factor = 0.15

# ================================END=================================

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)),
)
