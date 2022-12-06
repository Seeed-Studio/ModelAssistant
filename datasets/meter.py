import os
import math
import copy
from abc import ABCMeta

import cv2
import torch
import numpy as np
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset
from mmpose.datasets.builder import DATASETS
from mmcv.parallel import DataContainer as DC

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
class MeterData(Dataset, metaclass=ABCMeta):
    CLASSES = ('meter')

    def __init__(self,
                 data_root,
                 index_file,
                 pipeline,
                 format='xy',
                 test_mode=None):
        super(MeterData, self).__init__()

        self.data_root = check_file(data_root)

        self.transforms = Pose_Compose(
            pipeline, keypoint_params=A.KeypointParams(format))
        self.totensor = transforms.Compose([transforms.ToTensor()])

        with open(os.path.join(self.data_root, index_file), 'r') as f:
            self.lines = f.readlines()
        self.parse_ann()

    def __getitem__(self, item):
        ann = copy.deepcopy(self.ann_ls[item])
        img_file = ann['image_file']
        self.img = cv2.imread(img_file)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        h, w = self.img.shape[:-1]
        points = ann['keypoints']
        point_num = ann['point_num']
        landmark = []
        for i in range(point_num):
            landmark.append([points[i * 2], points[i * 2 + 1]])

        # if not self.test:
        while True:
            result = self.transforms(image=self.img, keypoints=landmark)
            if len(result['keypoints']) == point_num:
                break
        img, keypoints = self.totensor(result['image']), np.asarray(
            result['keypoints']).flatten()
        keypoints[::2] = keypoints[::2] / w
        keypoints[1::2] = keypoints[1::2] / h

        ann['img'] = img
        ann['keypoints'] = keypoints
        ann['image_file'] = DC(img_file, cpu_only=True)

        return ann

    def __len__(self):
        return len(self.ann_ls)

    def evaluate(self, results, **kwargs):
        return {
            'loss':
            torch.mean(
                torch.tensor([
                    i['loss'] for i in results if 'loss' in i.keys()
                ])).cpu().item()
        }

    def parse_ann(self):
        self.ann_ls = []
        for ann in self.lines:
            line = ann.strip().split()
            img_file = os.path.join(self.data_root, line[0])
            points = np.asarray(line[1:], dtype=np.float32)
            point_num = len(points) // 2

            self.ann_ls.append({
                'image_file': img_file,
                'keypoints': points,
                'point_num': point_num
            })
