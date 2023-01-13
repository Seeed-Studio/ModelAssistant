import os
import os.path as osp

import cv2
import torch
import json
import torchvision
import numpy as np
import albumentations as A
from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from mmdet.datasets.builder import DATASETS
from sklearn.metrics import confusion_matrix
from datasets.pipelines.pose_transform import Pose_Compose


@DATASETS.register_module()
class FomoDatasets(Dataset):

    def __init__(self,
                 data_root,
                 pipeline,
                 classes,
                 bbox_params: dict = dict(format='coco',
                                          label_fields=['class_labels']),
                 ann_file: str = None,
                 img_prefix: str = None,
                 test_mode=None) -> None:
        super().__init__()
        self.CLASSES = classes
        if not osp.isabs(img_prefix):
            img_dir = os.path.join(data_root, img_prefix)
        if not osp.isabs(ann_file):
            ann_file = os.path.join(data_root, ann_file)
        self.bbox_params = bbox_params
        self.transform = Pose_Compose(pipeline,
                                      bbox_params=A.BboxParams(**bbox_params))
        self.data = torchvision.datasets.CocoDetection(
            img_dir,
            ann_file,
        )
        self.totensor = ToTensor()
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, ann = self.data[index]
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        bboxes = []
        labels = []
        for annotation in ann:
            bboxes.append(annotation['bbox'])
            labels.append(annotation['category_id'])

        bboxes = np.array(bboxes)
        labels = np.array(labels)

        trans_param = {
            'image': image,
            'bboxes': bboxes,
            self.bbox_params['label_fields'][0]: labels
        }
        while True:
            result = self.transform(**trans_param)
            image_ = result['image']
            bboxes_ = result['bboxes']
            labels_ = result[self.bbox_params['label_fields'][0]]
            if len(np.array(bboxes_).flatten()) != (4 * len(bboxes)):
                continue
            else:
                image = image_
                bboxes = bboxes_
                labels = labels_
                break
        H, W, C = image.shape
        bbl = []
        for bbox, l in zip(bboxes, labels):

            bbl.append([
                0, l, (bbox[0] + (bbox[2] / 2)) / H,
                (bbox[1] + (bbox[3] / 2)) / W, bbox[2] / H, bbox[3] / W
            ])
        self.data
        return ToTensor()(image), torch.from_numpy(np.asarray(bbl))

    def get_ann_info(self, idx):
        img, ann = self.__getitem__[idx]
        return ann

    def bboxe2cell(self, bboxe, img_h, img_w, H, W):
        w = bboxe[0] + (bboxe[2] / 2)
        h = bboxe[1] + (bboxe[3] / 2)
        w = w / img_w
        h = h / img_h
        x = int(w * W)
        y = int(h * H)
        return (x, y)

    def build_target(self, preds, targets, img_h, img_w):
        B, H, W = preds.shape
        target_data = torch.zeros(size=(B, H, W), device=preds.device)
        target_data[..., 0] = 0
        bboxes = [i['bbox'] for i in targets]
        labels = [i['category_id'] for i in targets]

        bboxes = [
            self.bboxe2cell(bboxe, img_h, img_w, H, W) for bboxe in bboxes
        ]
        # print(labels)
        for bboxe, label in zip(bboxes, labels):
            target_data[0, bboxe[1], bboxe[0]] = label  #label

        return target_data

    def compute_FTP(self, pred, target):
        confusion = confusion_matrix(target.flatten().cpu().numpy(),
                                     pred.flatten().cpu().numpy(),
                                     labels=range(len(self.CLASSES) + 1))
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
        self.flag
        if fomo:  #just with here evaluate for fomo data
            eval_results = OrderedDict()
            tmp = []

            TP, FP, FN = [], [], []
            for idx, (pred, target) in enumerate(
                    zip(results['preds'], results['targets'])):
                B, H, W = pred.shape
                tp, fp, fn = self.compute_FTP(pred, target)
                mask = torch.eq(pred, target)
                acc = torch.sum(mask) / (H * W)
                tmp.append(acc)
                TP.append(tp)
                FP.append(fp)
                FN.append(fn)
                # fomo_show(pred,data['img_metas'].data['filename'],self.CLASSES,(512,512))
            P, R, F1 = self.computer_prf(sum(TP), sum(FP), sum(FN))
            # eval_results['Acc'] = torch.mean(torch.Tensor(tmp)).cpu().item()
            eval_results['P'] = P
            eval_results['R'] = R
            eval_results['F1'] = F1
            return eval_results
