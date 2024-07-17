_base_ = [
    'models/timm_resnet50.py', 'datasets/imagenet_bs32.py',
    'schedules/imagenet_bs256.py', '_base_/default_runtime.py'
]
