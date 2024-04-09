# Copyright (c) Seeed Tech Ltd. All rights reserved.
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2cocoxywh(x, coco_format: bool = False):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # top left x or center x
    y[:, 0] = x[:, 0] if coco_format else (x[:, 0] + x[:, 2]) / 2
    # top left y or center y
    y[:, 1] = x[:, 1] if coco_format else (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def NMS_FREE(
    bbox: Union[np.ndarray, torch.Tensor],
    classer: Union[np.ndarray, torch.Tensor],
    topk: int,
    bbox_format='xyxy',
    iou_thres=0.4,
    conf_thres=0.25,
):
    bbox = bbox if isinstance(bbox, torch.Tensor) else torch.from_numpy(bbox)
    classer = classer if isinstance(classer, torch.Tensor) else torch.from_numpy(classer)

    assert bbox.shape[0] == classer.shape[0]

    cls_mask = classer > conf_thres
    scores = classer[cls_mask]

    cls_idx = torch.nonzero(cls_mask)
    num_topk = min(topk, cls_idx.size(0))

    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = cls_idx[idxs[:num_topk]]
    keep_idxs, classer = topk_idxs.unbind(dim=1)

    bbox = bbox[keep_idxs]

    if bbox_format == 'xyxy':
        pass
    elif bbox_format == 'xywh':
        bbox = xywh2xyxy(bbox)

    pred = torch.cat((bbox, scores.view(-1, 1), classer[:, None]), 1)

    bbox, confiden = pred[:, :4], pred[:, 4]
    p = nms(bbox, confiden, iou_thres)

    res = pred[p]

    return res


def NMS(
    bbox: Union[np.ndarray, torch.Tensor],
    confiden: Union[np.ndarray, torch.Tensor],
    classer: Union[np.ndarray, torch.Tensor],
    bbox_format='xyxy',
    max_det=300,
    iou_thres=0.4,
    conf_thres=0.25,
):
    bbox = bbox if isinstance(bbox, torch.Tensor) else torch.from_numpy(bbox)
    confiden = confiden if isinstance(confiden, torch.Tensor) else torch.from_numpy(confiden)
    classer = classer if isinstance(classer, torch.Tensor) else torch.from_numpy(classer)

    assert bbox.shape[0] == confiden.shape[0] == classer.shape[0]

    conf_mask = confiden[0:] > conf_thres

    confiden = confiden[conf_mask]
    bbox = bbox[conf_mask]
    classer = classer[conf_mask]

    if bbox_format == 'xyxy':
        pass
    elif bbox_format == 'xywh':
        bbox = xywh2xyxy(bbox)

    pred = torch.cat((bbox, confiden.view(-1, 1), torch.argmax(classer, dim=1, keepdim=True)), 1)

    if pred.shape[0] > max_det:
        pred = pred[pred[:, 4].argsort(descending=True)[:max_det]]

    bbox, confiden = pred[:, :4], pred[:, 4]
    p = nms(bbox, confiden, iou_thres)

    res = pred[p]

    return res


def load_image(
    path: str,
    shape: Union[int, Tuple[int, int], None] = None,
    channels: Optional[int] = None,
    mode: str = 'RGB',
    normalized: bool = False,
    format: str = 'np',
) -> Union[np.ndarray, Image.Image]:
    assert format in ['np', 'pil'], ValueError

    img = cv2.imread(path)
    if shape:
        img = cv2.resize(img, shape[::-1])

    if mode and mode != 'BGR':
        img = cv2.cvtColor(img, getattr(cv2, 'COLOR_BGR2' + mode))
        if mode == 'GRAY' and channels and channels > 1:
            img = np.expand_dims(img, -1).repeat(channels, -1)

    if normalized:
        img = (img / 255).astype(np.float32)

    if format == 'pil':
        img = Image.fromarray(img)

    return img


if __name__ == '__main__':
    bbox = np.random.random((500, 4))
    conf = np.random.random((500))
    classes = np.random.random((500, 11))
    NMS(bbox=bbox, confiden=conf, classer=classes, bbox_format='xywh')
