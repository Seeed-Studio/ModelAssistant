# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
default_scope = "sscma"

# defaults input type image
input_type = "image"

from mmengine.hooks import (
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
    CheckpointHook,
    DistSamplerSeedHook,
)
from sscma.visualization.local_visualizer import PoseLocalVisualizer
from mmengine.visualization import LocalVisBackend, TensorboardVisBackend
from mmengine.runner import LogProcessor
from sscma.visualization import UniversalVisualizer

# ========================Suggested optional parameters========================
# RUNNING
# Model validation interval in epoch
val_interval = 5
# Model weight saving interval in epochs
save_interval = val_interval

# ================================END=================================
# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per epoch.
    checkpoint=dict(type=CheckpointHook, interval=5),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
    # visualization=dict(type=DetVisualizationHook)
)
# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type="SyncBuffersHook")
]

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=True,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)


# set visualizer
vis_backends = [dict(type=LocalVisBackend), dict(type=TensorboardVisBackend)]
visualizer = dict(type=PoseLocalVisualizer, vis_backends=vis_backends)


# set log level
log_level = "INFO"
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True, num_digits=6)


# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# file I/O backend
backend_args = dict(backend="local")

# training/validation/testing progress
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=val_interval)
val_cfg = dict()
test_cfg = dict()
