# Copyright (c) Seeed Tech Ltd. All rights reserved.
import torch
from mmdet.structures.det_data_sample import SampleList
from mmengine.structures.instance_data import InstanceData

from sscma.registry import MODELS

from .base import BasePseudoLabelCreator


@MODELS.register_module()
class LabelMatch(BasePseudoLabelCreator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.cls_thr = None
        self.cls_thr_ig = None

    def generate_pseudo_labels_online(
        self,
        teach_pred,
        student_sample: SampleList,
    ):
        if self.cls_thr is not None and self.cls_thr_ig is not None:
            for idx_sample, pred in enumerate(teach_pred):
                pseudo_bboxs, pseudo_label = [], []
                pseudo_bboxs_ig, pseudo_label_ig = [], []
                gt = InstanceData()
                ignore = InstanceData()
                for idx, label in enumerate(pred.pred_instances.labels):
                    if pred.pred_instances.scores[idx] > self.cls_thr[int(label)]:
                        pseudo_bboxs.append(pred.pred_instances.bboxes[idx])
                        pseudo_label.append(label)
                    elif pred.pred_instances.scores[idx] > self.cls_thr_ig[int(label)]:
                        pseudo_bboxs_ig.append(pred.pred_instances.bboxes[idx])
                        pseudo_label_ig.append(label)
                if len(pseudo_bboxs):
                    gt.bboxes = torch.concat([i.unsqueeze(0) for i in pseudo_bboxs], dim=0)
                    gt.labels = torch.concat([i.unsqueeze(0) for i in pseudo_label], dim=0)
                if len(pseudo_bboxs_ig):
                    ignore.labels = torch.concat([i.unsqueeze(0) for i in pseudo_label_ig], dim=0)
                    ignore.bboxes = torch.concat([i.unsqueeze(0) for i in pseudo_bboxs_ig], dim=0)
                student_sample[idx_sample].gt_instannces = gt
                student_sample[idx_sample].ignored_instances = ignore
