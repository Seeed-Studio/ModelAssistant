# checkpoint saving
checkpoint_config = dict(interval=5)
# config logging
log_config = dict(
    interval=150,
    hooks=[
        # dict(type='TextLoggerHook', ndigits=4),
        dict(type='TensorboardLoggerHook', ndigits=4)
    ])

# runtime settings
epochs=300
runner = dict(type='EpochBasedRunner', max_epochs=epochs)

dist_params = dict(backend='nccl')

log_level = 'INFO'

load_from = None
resume_from = None

workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 1

# set multi-process start method as `fork` to speed up the training
# mp_start_method = 'fork'

work_dir = './work_dirs'