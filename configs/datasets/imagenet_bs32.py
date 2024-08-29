from sscma.datasets import ImageNet
<<<<<<< Updated upstream
from sscma.datasets.transforms.loading import LoadImageFromFile
from sscma.datasets.transforms.processing import RandomResizedCrop, ResizeEdge, CenterCrop
from sscma.datasets.transforms.formatting import PackInputs
from sscma.datasets.transforms.transforms import RandomFlip

=======
from sscma.datasets.lancedb_datasets import LanceDataset
from sscma.datasets.transforms import (LoadImageFromFile,
									   RandomResizedCrop,
									   RandomFlip,
									   ResizeEdge,
									   CenterCrop,
									   PackInputs)
>>>>>>> Stashed changes
from mmengine.dataset import DefaultSampler
from sscma.evaluation import Accuracy

# dataset settings
dataset_type = LanceDataset


train_pipeline = [
    # dict(type=LoadImageFromFile,imdecode_backend='cv2'),
    dict(type=RandomResizedCrop, scale=224),
    dict(type=RandomFlip, prob=0.5, direction="horizontal"),
    dict(type=PackInputs),
]

test_pipeline = [
    # dict(type=LoadImageFromFile,imdecode_backend='cv2'),
    dict(type=ResizeEdge, scale=256, edge='short'),
    dict(type=CenterCrop, crop_size=224),
    dict(type=PackInputs),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root='/home/dq/datasets/emotion/',
        data_prefix=dict(img_path='train'),
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root='/home/dq/datasets/emotion/',
        data_prefix=dict(img_path='valid'),
        pipeline=test_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=False),
)
val_evaluator = dict(type=Accuracy, topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
