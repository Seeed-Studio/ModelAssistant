from typing import Optional, Sequence, Tuple, Union

import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from mmengine.model import BaseModule
from mmengine.model import normal_init, constant_init
from mmcv.cnn import is_norm

from edgelab.registry import LOSSES
from ..base.general import CBR


@MODELS.register_module()
class FomoHead(BaseModule):
    """
    The difference between the Fomo model head and the target detection model head 
    is that the output of this model only contains the probability of all categories, 
    and does not contain the value of xyhw
    """

    def __init__(
        self,
        input_channels: Union[Sequence[int], int],
        middle_channel: int = 48,
        num_classes: int = 20,
        act_cfg: str = 'ReLU6',
        loss_weight: Optional[Sequence[int]] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        loss_cls: Optional[dict] = dict(type='BCEWithLogitsLoss',
                                        reduction='mean'),
        loss_bg: Optional[dict] = dict(type='BCEWithLogitsLoss',
                                       reduction='mean'),
        init_cfg: Optional[dict] = dict(type='Normal', std=0.01)
    ) -> None:
        super(FomoHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.input_channels = input_channels if isinstance(
            input_channels, Sequence) else [input_channels]

        self.middle_channels = middle_channel[0] if isinstance(
            middle_channel, Sequence) else middle_channel
        self.act_cfg = act_cfg

        if loss_weight:
            for idx, w in enumerate(loss_weight):
                self.weight_cls[idx + 1] = w

        self.loss_bg = LOSSES.build(loss_bg)
        self.loss_cls = LOSSES.build(loss_cls)

        # Offset of the ground truth box
        self.posit_offset = torch.tensor(
            [[0, -1, 0], [0, -1, -1], [0, 0, -1], [0, 1, 0], [0, 1, 1],
             [0, 0, 1], [0, 1, -1], [0, -1, 1], [0, 0, 0]],
            dtype=torch.long)
        self._init_layers()

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(len(self.input_channels)):
            self.convs_bridge.append(
                CBR(self.input_channels[i],
                    self.middle_channels,
                    3,
                    1,
                    padding=1,
                    act=self.act_cfg))
            self.convs_pred.append(
                nn.Conv2d(self.middle_channels, self.num_attrib, 1, padding=0))

    def forward(self, x: Tuple[torch.Tensor, ...]):
        """
        Forward features from the upstream network.
        Args:
            x(tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        
        Returns:
            (tuple): Output the corresponding number of predicted feature 
                maps according to the output of the upstream network
        
        """
        assert len(x) == len(self.input_channels)

        result = []
        for i, feat in enumerate(x):
            feat = self.convs_bridge[i](feat)
            pred_map = self.convs_pred[i](feat)
            result.append(pred_map)

        return tuple(result)

    def loss(self, inputs: Tuple[torch.Tensor, ...], data_samples):
        pred = self.forward(inputs)
        return self.lossFunction(pred[0], data_samples[0].labels)

    def predict(self, features, data_samples, rescale=False):
        pred = self.forward(features)
        return [
            InstanceData(pred=pred[0],
                         labels=self.build_target(pred[0].permute(0, 2, 3, 1),
                                                  data_samples[0].labels))
        ]

    def lossFunction(self, pred_maps: torch.Tensor, target: torch.Tensor):
        """ Calculate the loss of the model 
        Args:
            preds(torch.Tensor): Model Predicted Output
            target(torch.Tensor): The target feature map constructed according to the 
                size of the feature map output by the model
                
        Returns:
            (dict): The model loss value in the training phase and the evaluation index 
                of the model
        """
        preds = pred_maps.permute(0, 2, 3, 1)
        B, H, W, C = preds.shape
        # pos_weights
        weight = torch.zeros(self.num_attrib, device=preds.device)
        weight[0] = 1
        self.weight_mask = torch.tile(weight, (H, W, 1))
        # Get the ground truth box that fits the fomo model
        data = self.build_target(preds, target)
        # background loss
        bg_loss = self.loss_bg(
            preds,
            data,
        )
        bg_loss *= self.weight_mask
        # no background loss
        cls_loss = self.loss_cls(
            preds,
            data,
        )
        cls_loss *= 1.0 - self.weight_mask
        # avg loss
        loss = torch.mean(cls_loss + bg_loss)
        # get p,r,f1
        P, R, F1 = self.get_pricsion_recall_f1(preds, data)
        return dict(loss=loss,
                    fgnd=cls_loss,
                    bgnd=bg_loss,
                    P=torch.Tensor([P]),
                    R=torch.Tensor([R]),
                    F1=torch.Tensor([F1]))

    def get_pricsion_recall_f1(self, preds: torch.Tensor,
                               target: torch.Tensor):
        """ 
        Calculate the evaluation index of model prediction 
        according to the prediction result of the model and the target feature map.
        
        Args:
            preds(torch.Tensor): Model Predicted Output
            target(torch.Tensor): The target feature map constructed according to the 
                size of the feature map output by the model
                
        Returns:
            P: Precision
            R: Recall
            F1: F1
        """
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
                # Avoid index out of bounds
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
                                     labels=range(self.num_attrib))
        # Calculate the value of P、R、F1 based on the confusion matrix
        tn = confusion[0, 0]
        tp = np.diagonal(confusion).sum() - tn
        fn = np.tril(confusion, k=-1).sum()
        fp = np.triu(confusion, k=1).sum()
        # Denominator cannot be zero
        if tp + fp == 0 or tp + fn == 0:
            return 0.0, 0.0, 0.0
        # calculate
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * (p * r) / (p + r) if p + r != 0 else 0

        return p, r, f1

    def build_target(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        The target feature map constructed according to the size 
        of the feature map output by the model
        """
        B, H, W, C = preds.shape
        target_data = torch.zeros(size=(B, H, W, C), device=preds.device)
        target_data[..., 0] = 1

        for i in targets:
            h, w = int(i[3].item() * H), int(i[2].item() * W)
            target_data[int(i[0]), h, w, 0] = 0  # background
            target_data[int(i[0]), h, w, int(i[1])] = 1  #label

        return target_data

    @property
    def num_attrib(self):
        """ The number of classifications the model needs to classify (including background)
            Return:
                (int): Add one to the number of categories
        """
        return self.num_classes + 1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
