# checkpoint saving
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=150,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1500)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs'
load_from = None
resume_from = None
workflow = [('train', 1)]