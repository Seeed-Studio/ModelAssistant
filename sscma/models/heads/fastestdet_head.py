# Copyright (c) Seeed Tech Ltd. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torchvision
from mmengine.model import BaseModule, constant_init, normal_init

from sscma.models.base import is_norm
from sscma.registry import MODELS

from ..base.general import CBR
from ..utils.metrics import bbox_iou


@MODELS.register_module()
class Fastest_Head(BaseModule):
    def __init__(
        self,
        input_channels,
        num_classes=80,
        loss_conf: dict = dict(type='SmoothL1Loss', reduction='none'),
        loss_cls: dict = dict(type='NLLLoss'),
        train_cfg: dict = None,
        test_cfg: dict = None,
        init_cfg: Optional[dict] = dict(type='Normal', std=0.01),
    ) -> None:
        super(Fastest_Head, self).__init__(init_cfg)

        self.loss_cls = nn.SmoothL1Loss(reduction='none')
        self.loss_conf = nn.NLLLoss()

        self.num_classes = num_classes

        self.conv1 = CBR(input_channels, input_channels, 1, 1, padding=0)

        self.obj_layers = self._make_layer(input_channels, 1)
        self.reg_layers = self._make_layer(input_channels, 4)
        self.cls_layers = self._make_layer(input_channels, num_classes)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, inp, 5, 1, 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        x = self.conv1(x)
        obj = self.sigmoid(self.obj_layers(x))
        reg = self.reg_layers(x)
        cls = self.softmax(self.cls_layers(x))
        return torch.concat((obj, reg, cls), dim=1)

    def forward_train(
        self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None, proposal_cfg=None, **kwargs
    ):
        results = self(x)
        loss = self.loss(
            results, gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_bbox_ignore=gt_bboxes_ignore, img_metas=img_metas
        )
        return loss

    def simple_test(self, img, img_metas, rescale=False):
        result = self(img)
        results_list = self.handle_preds(result, result.device, img_metas[0][0]['ori_shape'][:2])
        return results_list

    def loss(self, pred_maps, gt_bboxes, gt_labels, img_metas, gt_bbox_ignore=None):
        target = self.merge_gt(gt_bboxes=gt_bboxes, gt_labels=gt_labels, img_metas=img_metas)
        gt_box, gt_cls, ps_index = self.build_target(pred_maps, target)

        ft = torch.cuda.FloatTensor if pred_maps[0].is_cuda else torch.Tensor
        cls_loss, iou_loss, obj_loss = ft([0]), ft([0]), ft([0])

        pred = pred_maps.permute(0, 2, 3, 1)
        pobj = pred[:, :, :, 0]
        preg = pred[:, :, :, 1:5]
        pcls = pred[:, :, :, 5:]

        N, H, W, C = pred.shape
        tobj = torch.zeros_like(pobj)
        factor = torch.ones_like(pobj) * 0.75

        if len(gt_box) > 0:
            # Calculate the detection box regression loss
            b, gx, gy = ps_index[0]
            ptbox = torch.ones((preg[b, gy, gx].shape)).to(pred_maps.device)
            ptbox[:, 0] = preg[b, gy, gx][:, 0].tanh() + gx
            ptbox[:, 1] = preg[b, gy, gx][:, 1].tanh() + gy
            ptbox[:, 2] = preg[b, gy, gx][:, 2].sigmoid() * W
            ptbox[:, 3] = preg[b, gy, gx][:, 3].sigmoid() * H

            # Calculate the detection frame IOU loss
            iou = bbox_iou(ptbox, gt_box[0], x1y1x2y2=False)
            # Filter
            f = iou > iou.mean()
            b, gy, gx = b[f], gy[f], gx[f]

            # computer iou loss
            iou = iou[f]
            iou_loss = (1.0 - iou).mean()

            # Calculate the target category classification branch loss
            ps = torch.log(pcls[b, gy, gx])
            cls_loss = self.loss_cls(ps, gt_cls[0][f])

            # iou aware
            tobj[b, gy, gx] = iou.float()
            # Count the number of positive samples for each image
            n = torch.bincount(b)
            factor[b, gy, gx] = (1.0 / (n[b] / (H * W))) * 0.25

        obj_loss = (
            self.loss_conf(
                pobj,
                tobj,
            )
            * factor
        ).mean()

        loss = (iou_loss * 8) + (obj_loss * 16) + cls_loss

        return dict(loss=loss, iou_loss=iou_loss, cls_loss=cls_loss, obj_loss=obj_loss)

    def build_target(self, preds, targets):
        N, C, H, W = preds.shape

        gt_box, gt_cls, ps_index = [], [], []
        # The four vertices of each grid are the reference points where the center point of the box will return
        quadrant = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], device=preds.device)
        if targets.shape[0] > 0:
            # Map the coordinates onto the feature map scale
            scale = torch.ones(6).to(preds.device)

            scale[2:] = torch.tensor(preds.shape)[[3, 2, 3, 2]]
            gt = targets * scale

            # Extended Dimensions Copy Data
            gt = gt.repeat(4, 1, 1)

            # Filter out-of-bounds coordinates
            quadrant = quadrant.repeat(gt.size(1), 1, 1).permute(1, 0, 2)
            gij = gt[..., 2:4].long() + quadrant
            j = torch.where(gij < H, gij, 0).min(dim=-1)[0] > 0

            # foreground position index
            gi, gj = gij[j].T
            batch_index = gt[..., 0].long()[j]
            ps_index.append((batch_index, gi, gj))

            # foreground box
            gbox = gt[..., 2:][j]
            gt_box.append(gbox)

            # category of foreground
            gt_cls.append(gt[..., 1].long()[j])

        return gt_box, gt_cls, ps_index

    def merge_gt(self, gt_bboxes, gt_labels, img_metas):
        target = []

        max_size = max(img_metas[0]['img_shape'][:2])
        for idx, (labels, bboxes) in enumerate(zip(gt_labels, gt_bboxes)):
            bboxes = bboxes / max_size
            bb = torch.zeros_like(bboxes, device=bboxes.device)
            bb[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / 2
            bb[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / 2
            bb[..., 2] = bboxes[..., 2] - bboxes[..., 0]
            bb[..., 3] = bboxes[..., 3] - bboxes[..., 1]

            a = bb.shape[0]

            labels = labels.reshape((a, 1))
            z0 = torch.zeros((a, 1), device=bboxes.device) + idx
            bb = torch.concat((labels, bb), 1)
            gt = torch.concat((z0, bb), axis=1)
            target.append(gt)

        target = torch.concat(target, 0)

        return target

    def handle_preds(self, preds, device, shape, conf_thresh=0.25, nms_thresh=0.45):
        total_bboxes, output_bboxes = [], []
        # Convert the feature map to the coordinates of the detection box
        N, C, H, W = preds.shape
        bboxes = torch.zeros((N, H, W, 6))

        pred = preds.permute(0, 2, 3, 1)
        # confidence
        pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
        # bboxes
        preg = pred[:, :, :, 1:5]
        # cls
        pcls = pred[:, :, :, 5:]

        # bboxe confidence
        bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
        bboxes[..., 5] = pcls.argmax(dim=-1)

        # bboxes coordinate

        gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)])
        bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid()
        bcx = (preg[..., 0].tanh() + gx.to(device)) / W
        bcy = (preg[..., 1].tanh() + gy.to(device)) / H

        # cx,cy,w,h = > x1,y1,x2,y1
        x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
        x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

        bboxes[..., 0], bboxes[..., 1] = x1, y1
        bboxes[..., 2], bboxes[..., 3] = x2, y2
        bboxes = bboxes.reshape(N, H * W, 6)
        total_bboxes.append(bboxes)

        batch_bboxes = torch.cat(total_bboxes, 1)

        # NMS
        for batch in batch_bboxes:
            output, temp = [], []
            coord, scores, idxs = [], [], []
            # conf filter
            mask = batch[:, 4] > conf_thresh
            if not torch.any(mask):
                output_bboxes.append((torch.Tensor([]), torch.Tensor([0])))
                continue

            for bbox in batch[mask]:
                obj_score = bbox[4]
                category = bbox[5]
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[2], bbox[3]
                scores.append([obj_score])
                idxs.append([category])
                coord.append([x1, y1, x2, y2])
                temp.append([x1, y1, x2, y2, obj_score, category])
            # Torchvision NMS
            if len(coord) > 0:
                coord = torch.Tensor(coord).to(device)
                idxs = torch.Tensor(idxs).squeeze(1).to(device)
                scores = torch.Tensor(scores).squeeze(1).to(device)
                keep = torchvision.ops.batched_nms(coord, scores, idxs, nms_thresh)
                for i in keep:
                    output.append(temp[i])
            output_bboxes.append(
                (
                    torch.Tensor(output)[..., :5] * torch.Tensor([shape[1], shape[0], shape[1], shape[0], 1]),
                    torch.Tensor(output)[..., 5],
                )
            )

        return output_bboxes

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

    @property
    def num_attrib(self):
        return 5 + self.num_classes
