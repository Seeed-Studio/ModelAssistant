# copyright Copyright (c) Seeed Technology Co.,Ltd.
_base_ = '../_base_/default_runtime_cls.py'

# defaults input type image
input_type = 'sensor'

# ========================Suggested optional parameters========================
# RUNNING
# Model validation interval in epoch
val_interval = 5
# Model weight saving interval in epochs
save_interval = val_interval

# defaults to use registries in mmpretrain
default_scope = 'sscma'
# ================================END=================================
# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),
    # print log every 100 iterations.
    logger=dict(type='TextLoggerHook', interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', save_best='auto', interval=save_interval),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # validation results visualization, set True to enable it.
    visualization=dict(type='sscma.SensorVisualizationHook', enable=False),
)
