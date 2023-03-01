import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import BACKBONES


@BACKBONES.register_module()
class AxesNet(nn.Module):

    def __init__(self,
                 num_axes=3,
                 frequency=62.5,
                 duration=1,
                 out_channels=256,
                 ):
        super().__init__()

        self.intput_feature = num_axes * int(frequency) * duration
        liner_feature = self.liner_feature_fit()
        self.fc1 = nn.Linear(in_features=self.intput_feature,
                             out_features=out_channels, bias=True)
        self.fc2 = nn.Linear(in_features=liner_feature,
                             out_features=liner_feature, bias=True)
        self.fc3 = nn.Linear(
            in_features=liner_feature, out_features=out_channels, bias=True)

    def liner_feature_fit(self):

        return (int(self.intput_feature / 256) + 1) * 256

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


if __name__ == '__main__':
    pass
