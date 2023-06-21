default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='edgelab.TextLoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook')
    )

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


vis_backends = [dict(type='LocalVisBackend'),
                # dict(type='WandbVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='FomoLocalVisualizer', vis_backends=vis_backends, name='visualizer')


log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

train_cfg = dict(by_epoch=True,max_epochs=300)
val_cfg = dict()
test_cfg = dict()

