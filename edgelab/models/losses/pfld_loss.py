import torch
import torch.nn as nn
from mmpose.models.builder import LOSSES


@LOSSES.register_module()
class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, landmarks, landmark_gt):
        # angle_loss = torch.sum(1-torch.cos((angle-angle_gt)),axis=0)
        landmark_gt = torch.from_numpy(landmark_gt)
        l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)

        return torch.mean(l2_distant)
