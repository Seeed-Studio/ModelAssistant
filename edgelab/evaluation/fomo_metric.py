from typing import Optional

from sklearn.metrics import confusion_matrix
import numpy as np
import torch
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class FomoMetric(BaseMetric):

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        self.posit_offset = torch.tensor(
            [[0, -1, 0], [0, -1, -1], [0, 0, -1], [0, 1, 0], [0, 1, 1],
             [0, 0, 1], [0, 1, -1], [0, -1, 1], [0, 0, 0]],
            dtype=torch.long)

    def compute_ftp(self, preds, target):
        preds, target = preds.to(torch.device('cpu')), target.to(
            torch.device('cpu'))
        preds = torch.softmax(preds, dim=-1)
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
                if torch.any(site < 0) or torch.any(site > 11):
                    continue
                # The prediction is considered to be correct if it is near the ground truth box
                if site in preds_index and preds_max[site.chunk(
                        3)] == target_max[ti.chunk(3)]:
                    preds_max[site.chunk(3)] = target_max[ti.chunk(3)]
                    target_max[site.chunk(3)] = target_max[ti.chunk(3)]
        # Calculate the confusion matrix
        confusion = confusion_matrix(target_max.flatten().cpu().numpy(),
                                     preds_max.flatten().cpu().numpy(),
                                     labels=range(preds.shape[-1]))
        # Calculate the value of P、R、F1 based on the confusion matrix
        tn = confusion[0, 0]
        tp = np.diagonal(confusion).sum() - tn
        fn = np.tril(confusion, k=-1).sum()
        fp = np.triu(confusion, k=1).sum()
        return tp, fp, fn

    def computer_prf(self, tp, fp, fn):
        # Denominator cannot be zero
        if tp + fp == 0 or tp + fn == 0:
            return 0.0, 0.0, 0.0
        # calculate
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * (p * r) / (p + r) if p + r != 0 else 0

        return p, r, f1

    def process(self, data_batch, data_samples) -> None:
        tmp = TP = FP = FN = []
        pred, target = data_samples[0], data_samples[1]
        pred = pred.permute(0, 2, 3, 1)
        B, H, W, C = target.shape
        tp, fp, fn = self.compute_ftp(pred, target)
        mask = torch.eq(pred, target)
        acc = torch.sum(mask) / (H * W)
        tmp.append(acc)
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
        self.results.append(dict(tp=tp, fp=fp, fn=fn))

    def compute_metrics(self, results: list) -> dict:
        tp = sum([i['tp'] for i in results])
        fp = sum([i['fp'] for i in results])
        fn = sum([i['fn'] for i in results])
        P, R, F1 = self.computer_prf(tp, fp, fn)
        return dict(
            P=P,
            R=R,
            F1=F1,
        )
