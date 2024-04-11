# copyright Copyright (c) Seeed Technology Co.,Ltd.
from typing import Any, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS


def pose_acc(pred, target, hw, th=10):
    h = hw[0] if isinstance(hw[0], int) else int(hw[0][0])
    w = hw[1] if isinstance(hw[1], int) else int(hw[1][0])
    pred[:, 0::2] = pred[:, 0::2] * w  # TypeError: unsupported operand type(s) for *: 'generator' and 'int'
    pred[:, 1::2] = pred[:, 1::2] * h
    pred[pred < 0] = 0

    target[:, 0::2] = target[:, 0::2] * w
    target[:, 1::2] = target[:, 1::2] * h

    th = th
    acc = []
    for p, t in zip(pred, target):
        distans = ((t[0] - p[0]) ** 2 + (t[1] - p[1]) ** 2) ** 0.5
        if distans > th:
            acc.append(0)
        elif distans > 1:
            acc.append((th - distans) / (th - 1))
        else:
            acc.append(1)
    return sum(acc) / len(acc)


@METRICS.register_module()
class PointMetric(BaseMetric):
    def __init__(self, collect_device: str = 'cpu', prefix: Optional[str] = 'keypoint') -> None:
        super().__init__(collect_device, prefix)

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        target = data_batch['data_samples']['keypoints']
        size = data_batch['data_samples']['hw']  # .cpu().numpy()
        result = np.array([i if isinstance(i, np.ndarray) else i.cpu().numpy() for i in data_samples[0]['results']])

        result = result if len(result.shape) == 2 else result[None, :]  # onnx shape(2,), tflite shape(1,2)
        acc = pose_acc(result.copy(), target, size)
        self.results.append({'Acc': acc, 'pred': result, 'image_file': data_batch['data_samples']['image_file']})

    def compute_metrics(self, results: list) -> dict:
        return {'Acc': sum([i['Acc'] for i in results]) / len(results)}
