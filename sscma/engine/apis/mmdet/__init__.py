# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from .test import (
    collect_results_cpu,
    collect_results_gpu,
    multi_gpu_test,
    single_gpu_test_fomo,
    single_gpu_test_mmcls,
)
from .train import auto_scale_lr, init_random_seed, set_random_seed, train_detector

__all__ = [
    'single_gpu_test_mmcls',
    'single_gpu_test_fomo',
    'init_random_seed',
    'set_random_seed',
    'auto_scale_lr',
    'train_detector',
    'multi_gpu_test',
    'collect_results_cpu',
    'collect_results_gpu',
]
