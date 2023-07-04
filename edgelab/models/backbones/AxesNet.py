import torch
import torch.nn as nn
import torch.nn.functional as F
from edgelab.registry import MODELS


@MODELS.register_module()
class AxesNet(nn.Module):

    def __init__(self,
                 num_axes=3,   # axes number
                 frequency=62.5,  # sample frequency
                 window=1000,  # window size
                 num_classes=-1
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.intput_feature = num_axes * int(frequency * window / 1000)
        liner_feature = self.liner_feature_fit()
        self.fc1 = nn.Linear(in_features=self.intput_feature,
                             out_features=liner_feature, bias=True)
        self.fc2 = nn.Linear(
            in_features=liner_feature, out_features=liner_feature, bias=True)

        if self.num_classes > 0:
            self.classifier = nn.Linear(in_features=liner_feature, out_features=num_classes, bias=True)


    def liner_feature_fit(self):

        return (int(self.intput_feature / 1024) + 1) * 256

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self.num_classes > 0:
            x = self.classifier(x)
            
        return (x, )


if __name__ == '__main__':
    pass
