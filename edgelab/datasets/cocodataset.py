import json
import os.path as osp
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from mmdet.datasets.coco import CocoDataset
from sklearn.metrics import confusion_matrix

from edgelab.registry import DATASETS

from .utils.download import check_file


@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):
    METAINFO = {
        'classes': (),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 70),
            (0, 0, 192),
            (250, 170, 30),
            (100, 170, 30),
            (220, 220, 0),
            (175, 116, 175),
            (250, 0, 30),
            (165, 42, 42),
            (255, 77, 255),
            (0, 226, 252),
            (182, 182, 255),
            (0, 82, 0),
            (120, 166, 157),
            (110, 76, 0),
            (174, 57, 255),
            (199, 100, 0),
            (72, 0, 118),
            (255, 179, 240),
            (0, 125, 92),
            (209, 0, 151),
            (188, 208, 182),
            (0, 220, 176),
            (255, 99, 164),
            (92, 0, 73),
            (133, 129, 255),
            (78, 180, 255),
            (0, 228, 0),
            (174, 255, 243),
            (45, 89, 255),
            (134, 134, 103),
            (145, 148, 174),
            (255, 208, 186),
            (197, 226, 255),
            (171, 134, 1),
            (109, 63, 54),
            (207, 138, 255),
            (151, 0, 95),
            (9, 80, 61),
            (84, 105, 51),
            (74, 65, 105),
            (166, 196, 102),
            (208, 195, 210),
            (255, 109, 65),
            (0, 143, 149),
            (179, 0, 194),
            (209, 99, 106),
            (5, 121, 0),
            (227, 255, 205),
            (147, 186, 208),
            (153, 69, 1),
            (3, 95, 161),
            (163, 255, 0),
            (119, 0, 170),
            (0, 182, 199),
            (0, 165, 120),
            (183, 130, 88),
            (95, 32, 0),
            (130, 114, 135),
            (110, 129, 133),
            (166, 74, 118),
            (219, 142, 185),
            (79, 210, 114),
            (178, 90, 62),
            (65, 70, 15),
            (127, 167, 115),
            (59, 105, 106),
            (142, 108, 45),
            (196, 172, 0),
            (95, 54, 80),
            (128, 76, 255),
            (201, 57, 1),
            (246, 0, 122),
            (191, 162, 208),
        ],
    }

    def __init__(
        self,
        ann_file: str = '',
        metainfo: Optional[dict] = None,
        data_root=None,
        data_prefix: dict = dict(img_path=''),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
        filter_supercat: bool = True,
        file_client_args: Optional[dict] = dict(backend='disk'),
        classes=None,
        **kwargs,
    ):
        if data_root:
            if not (osp.isabs(ann_file) and (osp.isabs(data_prefix['img']))):
                data_root = check_file(data_root, data_name='coco') if data_root else data_root
        if metainfo is None and not self.METAINFO['classes'] and not classes:
            if not osp.isabs(ann_file) and ann_file:
                self.ann_file = osp.join(data_root, ann_file)
            with open(self.ann_file, 'r') as f:
                data = json.load(f)
            if filter_supercat:
                categories = tuple(cat['name'] for cat in data['categories'] if cat['supercategory'] != 'none')
            else:
                categories = tuple(cat['name'] for cat in data['categories'])
            self.METAINFO['classes'] = categories
        elif classes:
            self.METAINFO['classes'] = classes

        super().__init__(
            ann_file,
            metainfo,
            data_root,
            data_prefix,
            filter_cfg,
            indices,
            serialize_data,
            pipeline,
            test_mode,
            lazy_init,
            max_refetch,
            **kwargs,
        )

    def bboxe2cell(self, bboxe, img_h, img_w, H, W):
        w = (bboxe[0] + bboxe[2]) / 2
        h = (bboxe[1] + bboxe[3]) / 2
        w = w / img_w
        h = h / img_h
        x = int(w * (W - 1))
        y = int(h * (H - 1))
        return (x, y)

    def build_target(self, preds, targets, img_h, img_w):
        B, H, W = preds.shape
        target_data = torch.zeros(size=(B, H, W), device=preds.device)
        target_data[..., 0] = 0
        bboxes = targets['bboxes']
        labels = targets['labels']

        bboxes = [self.bboxe2cell(bboxe, img_h, img_w, H, W) for bboxe in bboxes]

        for bboxe, label in zip(bboxes, labels):
            target_data[0, bboxe[1], bboxe[0]] = label + 1  # label

        return target_data

    def compute_FTP(self, pred, target):
        confusion = confusion_matrix(
            target.flatten().cpu().numpy(), pred.flatten().cpu().numpy(), labels=range(len(self.CLASSES) + 1)
        )
        tn = confusion[0, 0]
        tp = np.diagonal(confusion).sum() - tn
        fn = np.tril(confusion, k=-1).sum()
        fp = np.triu(confusion, k=1).sum()

        return tp, fp, fn

    def computer_prf(self, tp, fp, fn):
        if tp == 0 and fn == 0 and fp == 0:
            return 1.0, 1.0, 1.0

        p = 0.0 if (tp + fp == 0) else tp / (tp + fp)
        r = 0.0 if (tp + fn == 0) else tp / (tp + fn)
        f1 = 0.0 if (p + r == 0) else 2 * (p * r) / (p + r)
        return p, r, f1

    def evaluate(
        self,
        results,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        classwise=False,
        proposal_nums=...,
        iou_thrs=None,
        fomo=False,
        metric_items=None,
    ):
        if fomo:  # just with here evaluate for fomo data
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            eval_results = OrderedDict()
            tmp = []

            TP, FP, FN = [], [], []
            for idx, (pred, ann) in enumerate(zip(results, annotations)):
                data = self.__getitem__(idx)
                B, H, W = pred.shape
                img_h, img_w = data['img_metas'][0].data['ori_shape'][:2]
                target = self.build_target(pred, ann, img_h, img_w)
                tp, fp, fn = self.compute_FTP(pred, target)
                mask = torch.eq(pred, target)
                acc = torch.sum(mask) / (H * W)
                tmp.append(acc)
                TP.append(tp)
                FP.append(fp)
                FN.append(fn)
                # show_result(pred,data['img_metas'][0].data['filename'],self.CLASSES)
            P, R, F1 = self.computer_prf(sum(TP), sum(FP), sum(FN))
            eval_results['Acc'] = torch.mean(torch.Tensor(tmp)).cpu().item()
            eval_results['Acc'] = torch.mean(torch.Tensor(tmp)).cpu().item()
            eval_results['P'] = P
            eval_results['R'] = R
            eval_results['F1'] = F1
            return eval_results

        return super().evaluate(
            results, metric, logger, jsonfile_prefix, classwise, proposal_nums, iou_thrs, metric_items
        )


def show_result(result, img_path, classes):
    img = cv2.imread(img_path)
    H, W = img.shape[:-1]
    pred = result.cpu().numpy()
    mask = np.argwhere(pred > 0)
    for i in mask:
        b, h, w = i
        label = classes[pred[0, h, w] - 1]
        cv2.circle(
            img, (int(W / result[0].shape[1] * (w + 0.5)), int(H / result[0].shape[0] * (h + 0.5))), 5, (0, 0, 255), 1
        )
        cv2.putText(
            img,
            str(label),
            org=(int(W / result[0].shape[1] * w), int(H / result[0].shape[0] * h)),
            color=(255, 0, 0),
            fontScale=1,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        )
    cv2.imshow('img', img)
    cv2.waitKey(0)
