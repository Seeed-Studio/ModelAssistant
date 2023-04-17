from typing import Optional,Any,Sequence

import numpy as np

from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric

def pose_acc(pred, target, hw, th=10):
    h = hw[0] if isinstance(hw[0], int) else int(hw[0][0]) 
    w = hw[1] if isinstance(hw[1], int) else int(hw[1][0])
    pred[:, 0::2] = pred[:, 0::2] * w
    pred[:, 1::2] = pred[:, 1::2] * h
    pred[pred < 0] = 0

    target[:, 0::2] = target[:, 0::2] * w
    target[:, 1::2] = target[:, 1::2] * h

    th = th
    acc = []
    for p, t in zip(pred, target):
        distans = ((t[0] - p[0])**2 + (t[1] - p[1])**2)**0.5
        if distans > th:
            acc.append(0)
        elif distans > 1:
            acc.append((th - distans) / (th - 1))
        else:
            acc.append(1)
    return sum(acc) / len(acc)

@METRICS.register_module()
class PointMetric(BaseMetric):
    def __init__(self, collect_device: str = 'cpu', prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)
        
     
     
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        target = np.expand_dims(data_batch['keypoints'], axis=0)
        size = data_batch['hw']  #.cpu().numpy()
        result = np.array(data_samples.cpu())
        
        result = result if len(result.shape)==2 else result[None, :] # onnx shape(2,), tflite shape(1,2)
        acc = pose_acc(result.copy(), target, size)
        self.results.append({
            'Acc': acc,
            'pred': result,
            'image_file': data_batch['image_file'].data
        })
     
    
    def compute_metrics(self, results: list) -> dict:
        return super().compute_metrics(results)
    