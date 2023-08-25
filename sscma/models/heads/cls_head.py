import torch.nn as nn

from sscma.registry import MODELS


@MODELS.register_module()
class Audio_head(nn.Module):
    def __init__(self, in_channels, n_classes, drop=0.5):
        super(Audio_head, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, in_channels)
        self.fc1 = nn.Linear(in_channels, n_classes)
        self.dp = nn.Dropout(drop)

    def forward(self, x):
        return self.fc1(self.dp(self.fc(self.avg(x).flatten(1))))
