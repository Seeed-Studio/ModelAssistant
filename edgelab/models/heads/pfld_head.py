import torch
import torch.nn as nn
from mmpose.models.builder import HEADS

from ..base.general import CBR


@HEADS.register_module()
class PFLDhead(nn.Module):

    def __init__(self,
                 num_point=1,
                 input_channel=16,
                 feature_num=[32, 32]) -> None:
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = CBR(input_channel,
                         feature_num[0],
                         3,
                         2,
                         padding=1,
                         bias=False)
        self.conv2 = CBR(feature_num[0],
                         feature_num[1],
                         7,
                         1,
                         bias=False,
                         padding=0)
        # self.conv2 = nn.Conv2d(feature_num[0], feature_num[1], 7, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(input_channel + sum(feature_num), num_point * 2)

    def forward(self, x):
        x1 = self.avg_pool(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv1(x)
        x2 = self.avg_pool(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.conv2(x)
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks
