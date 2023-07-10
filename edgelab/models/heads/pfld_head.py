from typing import Sequence, Union

import torch
import torch.nn as nn

from edgelab.models.utils.computer_acc import pose_acc
from edgelab.registry import HEADS, LOSSES

from ..base.general import CBR


@HEADS.register_module()
class PFLDhead(nn.Module):
    """The head of the pfld model mainly uses convolution and global average
    pooling.

    Args:
        num_point: The model needs to predict the number of key points,
            and set the output of the model according to this value
        input_channel: The number of channels of the head input feature map
        feature_num: Number of channels in the middle feature map of the head
        act_cfg: Configuration of the activation function
        loss_cfg: Related configuration of model loss function
    """

    def __init__(
        self,
        num_point: int = 1,
        input_channel: int = 16,
        feature_num: Sequence[int] = [32, 32],
        act_cfg: Union[dict, str, None] = "ReLU",
        loss_cfg: dict = dict(type='PFLDLoss'),
    ) -> None:
        super().__init__()

        self.conv1 = CBR(input_channel, feature_num[0], 3, 2, padding=1, bias=False, act=act_cfg)
        self.conv2 = CBR(feature_num[0], feature_num[1], 2, 1, bias=False, padding=0, act=act_cfg)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_channel + sum(feature_num), num_point * 2)
        self.lossFunction = LOSSES.build(loss_cfg)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]

        x1 = self.avg_pool(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv1(x)
        x2 = self.avg_pool(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.conv2(x)
        x3 = self.avg_pool(x3)
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)

        landmarks = self.fc(multi_scale)

        return landmarks

    def loss(self, features, data_samples):
        preds = self.forward(features)
        labels = torch.as_tensor(data_samples['keypoints'], device=preds.device, dtype=torch.float32)
        loss = self.lossFunction(preds, labels)
        acc = pose_acc(preds.cpu().detach().numpy(), labels, data_samples['hw'])
        return {"loss": loss, "Acc": torch.as_tensor(acc, dtype=torch.float32)}

    def predict(self, features):
        return self.forward(features)
