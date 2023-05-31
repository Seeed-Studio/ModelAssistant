import copy
import os.path as osp
from collections import OrderedDict

import cv2
import torch
import numpy as np
from edgelab.registry import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmengine.fileio import load, get_local_path
from sklearn.metrics import confusion_matrix

from .utils.download import check_file


@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):

    def __init__(self,
                 ann_file,
                 metainfo=None,
                 data_root=None,
                 data_prefix='',
                 filter_cfg=None,
                 indices=None,
                 serialize_data: bool = True,
                 pipeline=[],
                 test_mode=False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 file_client_args=dict(backend='disk'),
                 classes=None,
                 **kwargs):
        if data_root:
            if not (osp.isabs(ann_file) and (osp.isabs(data_prefix['img']))):
                data_root = check_file(
                    data_root, data_name="coco") if data_root else data_root

        super().__init__(ann_file, metainfo, data_root, data_prefix,
                         filter_cfg, indices, serialize_data, pipeline,
                         test_mode, lazy_init, max_refetch, **kwargs)

        annotations = load(self.ann_file)
        print(annotations.keys())

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(self.ann_file,
                            backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.
            metainfo['classes'] if len(self.metainfo['classes']) else [
                cat['name'] for cat in self.coco.dataset['categories']
                if (cat['supercategory'] != "none")
            ])

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

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

        bboxes = [
            self.bboxe2cell(bboxe, img_h, img_w, H, W) for bboxe in bboxes
        ]

        for bboxe, label in zip(bboxes, labels):
            target_data[0, bboxe[1], bboxe[0]] = label + 1  #label

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
        if fomo:  #just with here evaluate for fomo data
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

        return super().evaluate(results, metric, logger, jsonfile_prefix,
                                classwise, proposal_nums, iou_thrs,
                                metric_items)


def show_result(result, img_path, classes):
    img = cv2.imread(img_path)
    H, W = img.shape[:-1]
    pred = result.cpu().numpy()
    mask = np.argwhere(pred > 0)
    for i in mask:
        b, h, w = i
        print(w, h)
        label = classes[pred[0, h, w] - 1]
        cv2.circle(img,
                   (int(W / result[0].shape[1] *
                        (w + 0.5)), int(H / result[0].shape[0] * (h + 0.5))),
                   5, (0, 0, 255), 1)
        cv2.putText(img,
                    str(label),
                    org=(int(W / result[0].shape[1] * w),
                         int(H / result[0].shape[0] * h)),
                    color=(255, 0, 0),
                    fontScale=1,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    cv2.imshow('img', img)
    cv2.waitKey(0)