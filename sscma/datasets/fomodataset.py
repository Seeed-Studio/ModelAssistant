# Copyright (c) Seeed Tech Ltd. All rights reserved.
import os
import os.path as osp

import albumentations as A
import numpy as np
import torch
import torchvision
from mmdet.datasets.coco import CocoDataset
from mmengine.registry import DATASETS
from sklearn.metrics import confusion_matrix

from .pipelines.composition import AlbCompose


@DATASETS.register_module()
class FomoDatasets(CocoDataset):
    def __init__(
        self,
        data_root,
        pipeline,
        classes=None,
        use_alb=True,
        bbox_params: dict = dict(format='coco', label_fields=['class_labels']),
        ann_file: str = None,
        img_prefix: str = None,
    ) -> None:
        super().__init__(ann_file=ann_file, data_root=data_root, pipeline=pipeline, data_prefix=dict(img='train/'))

        if not osp.isabs(img_prefix):
            img_dir = os.path.join(data_root, img_prefix)
        if not osp.isabs(ann_file):
            ann_file = os.path.join(data_root, ann_file)

        self.bbox_params = bbox_params

        self.transform = AlbCompose(pipeline, bbox_params=A.BboxParams(**bbox_params))
        # load data with coco format
        self.data = torchvision.datasets.CocoDetection(
            img_dir,
            ann_file,
        )

        self.parse_cats()
        # Offset of the ground truth box
        self.posit_offset = torch.tensor(
            [[0, -1, 0], [0, -1, -1], [0, 0, -1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, -1], [0, -1, 1], [0, 0, 0]],
            dtype=torch.long,
        )

        # TODO
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def parse_cats(self):
        """Parse dataset is roboflow."""
        self.roboflow = False
        self.CLASSES = []

        for key, value in self.data.coco.dataset['info'].items():
            if isinstance(value, str) and 'roboflow' in value:
                self.roboflow = True

        for key, value in self.data.coco.cats.items():
            if key == 0 and self.roboflow:
                continue
            self.CLASSES.append(value['name'])

    def __len__(self):
        """Return datasets len."""
        return len(self.data)

    def __getitem____(self, index):
        result = self.prepare_data(idx=index)
        return result

    def get_ann_info(self, idx):
        ann = self.__getitem__[idx]['target']
        return ann

    def bboxe2cell(self, bboxe, img_h, img_w, H, W):
        """Transform the bbox to ground cell."""
        w = bboxe[0] + (bboxe[2] / 2)
        h = bboxe[1] + (bboxe[3] / 2)
        w = w / img_w
        h = h / img_h
        x = int(w * W)
        y = int(h * H)
        return (x, y)

    def post_handle(self, preds, target):
        B, H, W, C = preds.shape
        assert (len(self.CLASSES) + 1) == C

        mask = torch.softmax(preds, dim=-1)
        values, indices = torch.max(mask, dim=-1)
        values_mask = np.argwhere(values.cpu().numpy() < 0.25)
        res = torch.argmax(mask, dim=-1)

        for i in values_mask:
            b, h, w = int(i[0].item()), int(i[1].item()), int(i[2].item())
            res[b, h, w] = 0

        return res, torch.argmax(self.build_target(preds, target), dim=-1)

    def build_target(self, preds, targets):
        B, H, W, C = preds.shape
        target_data = torch.zeros(size=(B, H, W, C), device=preds.device)
        target_data[..., 0] = 1
        for i in targets:
            h, w = int(i[3].item() * H), int(i[2].item() * W)
            target_data[int(i[0]), h, w, 0] = 0  # background
            target_data[int(i[0]), h, w, int(i[1])] = 1  # label

        return target_data

    def compute_ftp(self, preds, target):
        preds = torch.softmax(preds, dim=-1)
        B, C, H, W = preds.shape
        # Get the category id of each box
        target_max = torch.argmax(target, dim=-1)
        preds_max = torch.argmax(preds, dim=-1)
        # Get the index of the forecast for the non-background
        target_condition = torch.where(target_max > 0)
        preds_condition = torch.where(preds_max > 0)
        # splice index
        target_index = torch.stack(target_condition, dim=1)
        preds_index = torch.stack(preds_condition, dim=1)

        self.posit_offset = self.posit_offset.to(target.device)
        # Traversal compares predicted and ground truth boxes
        for ti in target_index:
            for po in self.posit_offset:
                site = ti + po
                # Avoid index out ofAvoid index out of bounds
                if torch.any(site < 0) or torch.any(site >= H):
                    continue
                # The prediction is considered to be correct if it is near the ground truth box
                if site in preds_index and preds_max[site.chunk(3)] == target_max[ti.chunk(3)]:
                    preds_max[site.chunk(3)] = target_max[ti.chunk(3)]
                    target_max[site.chunk(3)] = target_max[ti.chunk(3)]
        # Calculate the confusion matrix
        confusion = confusion_matrix(
            target_max.flatten().cpu().numpy(), preds_max.flatten().cpu().numpy(), labels=range(preds.shape[-1])
        )
        # Calculate the value of P, R, F1 based on the confusion matrix
        tn = confusion[0, 0]
        tp = np.diagonal(confusion).sum() - tn
        fn = np.tril(confusion, k=-1).sum()
        fp = np.triu(confusion, k=1).sum()
        return tp, fp, fn
