from sscma.datasets.coco import CocoDataset, coco_collate
from sscma.datasets.transforms.formatting import PackDetInputs
from sscma.datasets.transforms.processing import RandomResize
from sscma.datasets.transforms.loading import LoadAnnotations, LoadImageFromFile
from sscma.datasets.transforms.transforms import (
    CachedMixUp,
    CachedMosaic,
    Pad,
    RandomCrop,
    RandomFlip,
    Resize,
    YOLOXHSVRandomAug,
)

from mmengine.dataset.sampler import DefaultSampler
from time import perf_counter

import torch
from torch.utils.data import DataLoader
import tqdm

torch.random.manual_seed(0)


if __name__ == "__main__":
    BATCH_SIZE = 32

    print(f"\ntorch version: {torch.__version__}")

    for pin_memory in [False, True]:
        for numpy in [True]:

            train_pipeline = [
                dict(
                    type=LoadImageFromFile,
                    imdecode_backend="pillow",
                    backend_args=None,
                ),
                dict(type=LoadAnnotations, imdecode_backend="pillow", with_bbox=True),
                dict(type=CachedMosaic, img_scale=(640, 640), pad_val=114.0),
                dict(
                    type=RandomResize,
                    scale=(1280, 1280),
                    ratio_range=(0.1, 2.0),
                    resize_type=Resize,
                    keep_ratio=True,
                ),
                dict(type=RandomCrop, crop_size=(640, 640)),
                dict(type=YOLOXHSVRandomAug),
                dict(type=RandomFlip, prob=0.5),
                dict(type=Pad, size=(640, 640), pad_val=dict(img=(114, 114, 114))),
                dict(
                    type=CachedMixUp,
                    img_scale=(640, 640),
                    ratio_range=(1.0, 1.0),
                    max_cached_images=20,
                    pad_val=(114, 114, 114),
                ),
                dict(type=PackDetInputs),
            ]

            dataset = CocoDataset(
                data_root="/dataset/coco/",
                ann_file="annotations/instances_train2017.json",
                data_prefix=dict(img="train2017/"),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline,
            )
            sampler = DefaultSampler(dataset=dataset, shuffle=True)
            times = []

            print("\n" + "-" * 20 + f"\n{numpy=} {pin_memory=} \n")

            for workers in range(0, 13, 2):
                loader = DataLoader(
                    dataset=dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=workers,
                    collate_fn=coco_collate,
                    sampler=sampler,
                    pin_memory=pin_memory,
                )

                t = perf_counter()
                for x in tqdm.tqdm(loader, total=len(loader)):
                    continue
                t = perf_counter() - t
                times.append((workers, t))
                print(f"workers={workers}: {t:.2f}s")
