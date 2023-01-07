import os
import math
import torch
import numpy as np
from collections import OrderedDict

from mmdet.datasets.voc import VOCDataset
from mmdet.datasets.builder import DATASETS

from datasets.utils.download import check_file


@DATASETS.register_module()
class CustomVocdataset(VOCDataset):

    def __init__(self, **kwargs):
        kwargs['data_root'] = os.path.join(
            check_file(kwargs['data_root'], data_name='voc'), 'VOCdevkit',
            'VOC2012')

        super(CustomVocdataset, self).__init__(**kwargs)

    def bboxe2cell(self, bboxe, img_h, img_w, H, W):
        w = (bboxe[0] + bboxe[2]) / 2
        h = (bboxe[1] + bboxe[3]) / 2
        w = w / img_w
        h = h / img_h
        x = int(w * W)
        y = int(h * H)
        return (x, y)

    def build_target(self, preds, targets, img_h, img_w):
        B, H, W = preds.shape
        target_data = torch.zeros(size=(B, H, W), device=preds.device)
        target_data[..., 0] = 0
        bboxes = targets['bboxes']
        labels = targets['labels']

        bboxes = [
            self.bboxe2cell(bboxe, img_h, img_w, H, W) for bboxe in bboxes
        ]

        for bboxe, label in zip(bboxes, labels):
            target_data[0, bboxe[1], bboxe[0]] = label + 1  #label

        return target_data

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 fomo=False):
        if fomo:  #just with here evaluate for fomo data
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            eval_results = OrderedDict()
            tmp = []
            for idx, (pred, ann) in enumerate(zip(results, annotations)):
                data = self.__getitem__(idx)
                B, H, W = pred.shape
                img_h, img_w = data['img_metas'][0].data['ori_shape'][:2]
                target = self.build_target(pred, ann, img_h, img_w)
                mask = torch.eq(pred, target)
                acc = torch.sum(mask) / (H * W)
                tmp.append(acc)

            eval_results['Acc'] = torch.mean(torch.Tensor(tmp)).cpu().item()
            return eval_results

        else:  # object evaluate
            return super().evaluate(results,
                                    metric=metric,
                                    logger=logger,
                                    proposal_nums=proposal_nums,
                                    iou_thr=iou_thr,
                                    scale_ranges=scale_ranges)
