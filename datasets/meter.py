import os
import math

import cv2
import torch
import numpy as np
import albumentations as A

from mmpose.datasets.builder import DATASETS
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from datasets.utils.download import check_file
from datasets.pipelines.pose_transform import Pose_Compose


def calc_angle(x1, y1, x2, y2):
    x = (x1 - x2)
    y = (y1 - y2)
    z = math.sqrt(x * x + y * y)
    try:
        angle = math.acos(
            (z**2 + 1 - (x - 1)**2 - y**2) / (2 * z * 1)) / math.pi * 180
    except:
        angle = 0

    if y < 0:
        angle = 360 - angle

    return angle


@DATASETS.register_module()
class MeterData(Dataset):
    CLASSES = ('meter')

    def __init__(self,
                 data_root,
                 index_file,
                 pipeline,
                 format='xy',
                 test_mode=None):
        super(MeterData, self).__init__()

        self.data_root = check_file(data_root)

        self.test = test_mode
        self.pipeline = Pose_Compose(pipeline,
                                     keypoint_params=A.KeypointParams(format))
        self.test_trans = transforms.Compose([transforms.ToTensor()])

        with open(os.path.join(self.data_root, index_file), 'r') as f:
            self.lines = f.readlines()
        self.flag = np.zeros(self.__len__(), dtype=np.uint8)

    def __getitem__(self, item):
        self.line = self.lines[item].strip().split()
        img_file = os.path.join(self.data_root, self.line[0])
        img_file = str(img_file)
        self.img = cv2.imread(img_file)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        w = self.img.shape[1]
        points = np.asarray(self.line[1:], dtype=np.float32)
        point_num = len(points) // 2
        landmark = []
        for i in range(point_num):
            landmark.append([points[i * 2], points[i * 2 + 1]])

        if self.test:
            img, label = self.test_trans(
                self.img), np.asarray(landmark).flatten() / w
        else:
            while True:
                result = self.pipeline(image=self.img, keypoints=landmark)
                if len(result['keypoints']) < point_num:
                    continue
                else:
                    break
            img, label = self.test_trans(
                result['image']), np.asarray(result['keypoints']).flatten() / w

        return {'img': img, 'img_metas': label}

    def __len__(self):
        return len(self.lines)

    def evaluate(self, results, **kwargs):
        return {
            'loss':
            torch.mean(torch.tensor([i['loss']
                                     for i in results])).cpu().item()
        }
