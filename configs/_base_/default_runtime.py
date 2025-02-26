# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.visualization import LocalVisBackend, TensorboardVisBackend
from mmengine.runner import LogProcessor
from sscma.visualization import UniversalVisualizer

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
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=True,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
vis_backends = [dict(type=LocalVisBackend), dict(type=TensorboardVisBackend)]
visualizer = dict(type=UniversalVisualizer, vis_backends=vis_backends)

# set log level
log_level = "INFO"
log_processor = dict(type=LogProcessor, window_size=50, by_epoch=True)

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# Do not need to specify default_scope with new config. Therefore set it to
# None to avoid BC-breaking.
default_scope = None

#whether to dump the config file when starting the script
dump_config = False