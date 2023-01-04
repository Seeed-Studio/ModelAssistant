import os.path as osp
from collections import OrderedDict

import torch
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

from datasets.utils.download import check_file


@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 seg_suffix='.png',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):
        if data_root:
            if not (osp.isabs(ann_file) and
                    (osp.isabs(img_prefix) or osp.isabs(seg_prefix))):
                data_root = check_file(
                    data_root, data_name="coco") if data_root else data_root

        super().__init__(ann_file, pipeline, classes, data_root, img_prefix,
                         seg_prefix, seg_suffix, proposal_file, test_mode,
                         filter_empty_gt, file_client_args)
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
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=...,
                 iou_thrs=None,
                 fomo=False,
                 metric_items=None):
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

        return super().evaluate(results, metric, logger, jsonfile_prefix,
                                classwise, proposal_nums, iou_thrs,
                                metric_items)
