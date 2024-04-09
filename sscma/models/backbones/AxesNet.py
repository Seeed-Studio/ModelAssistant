# Copyright (c) Seeed Tech Ltd. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from sscma.registry import MODELS


@MODELS.register_module()
class AxesNet(nn.Module):
    def __init__(self, num_axes=3, window_size=80, num_classes=-1):  # axes number  # sample frequency  # window size
        super().__init__()
        self.num_classes = num_classes
        self.intput_feature = int(num_axes * window_size)
        liner_feature = self.liner_feature_fit()
        self.fc1 = nn.Linear(in_features=self.intput_feature, out_features=liner_feature, bias=True)
        self.fc2 = nn.Linear(in_features=liner_feature, out_features=liner_feature, bias=True)

        if self.num_classes > 0:
            self.classifier = nn.Linear(in_features=liner_feature, out_features=num_classes, bias=True)

    def liner_feature_fit(self):
        return (int(self.intput_feature / 1024) + 1) * 256

    def forward(self, x):
        x = x[0] if isinstance(x, list) else x
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        if self.num_classes > 0:
            x = self.classifier(x)

        return (x,)


if __name__ == '__main__':
    pass
