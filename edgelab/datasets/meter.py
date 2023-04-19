import os
import math
import copy
import json
import os.path as osp
from abc import ABCMeta

import cv2
import torch
import numpy as np
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset
from edgelab.registry import DATASETS

from .utils.download import check_file
from .pipelines.pose_transform import Pose_Compose


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
                 index_file: str,
                 img_dir: str = None,
                 pipeline=None,
                 format='xy',
                 test_mode=None):
        super(MeterData, self).__init__()

        self.data_root = check_file(data_root)
        self.img_dir = img_dir  #todo

        if img_dir and not osp.isabs(img_dir) and self.data_root:
            self.img_dir = osp.join(self.data_root, img_dir)
        if not osp.isabs(index_file) and self.data_root:
            index_file = osp.join(self.data_root, index_file)

        if osp.isdir(index_file):
            file_ls = os.listdir(index_file)
            file_ls = [osp.join(index_file, i) for i in file_ls]

            self.parse_jsons(file_ls)
        else:
            if index_file.endswith('txt'):
                with open(index_file, 'r') as f:
                    self.lines = f.readlines()
                self.parse_txt()
            elif index_file.endswith('json'):
                self.parse_json(index_file)

        self.transforms = Pose_Compose(
            pipeline, keypoint_params=A.KeypointParams(format))
        self.totensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item) -> dict:
        ann = copy.deepcopy(self.ann_ls[item])
        img_file = ann['image_file']
        self.img = cv2.imread(img_file)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        points = ann['keypoints']
        point_num = ann['point_num']
        landmark = []
        for i in range(point_num):
            landmark.append([points[i * 2], points[i * 2 + 1]])
        while True:
            result = self.transforms(image=self.img, keypoints=landmark)
            if len(result['keypoints']) == point_num:
                break
        img, keypoints = self.totensor(result['image']), np.asarray(
            result['keypoints']).flatten()
        h, w = img.shape[1:]
        keypoints[::2] = keypoints[::2] / w
        keypoints[1::2] = keypoints[1::2] / h

        ann['img'] = img
        ann['keypoints'] = keypoints
        ann['image_file'] = img_file
        ann['hw'] = [h, w]

        return ann

    def __len__(self) -> int:
        return len(self.ann_ls)

    def evaluate(self, results, **kwargs):
        return {
            'loss':
            torch.mean(
                torch.tensor([
                    i['loss'] for i in results if 'loss' in i.keys()
                ])).cpu().item(),
            'Acc':
            torch.mean(
                torch.tensor([i['Acc'] for i in results
                              if 'Acc' in i.keys()])).cpu().item()
        }

    def parse_json(self, json_path) -> None:
        pass  #todo

    def parse_jsons(self, json_ls) -> None:
        self.ann_ls = []
        for js in json_ls:

            tmp = {}
            point = []
            file = json.load(open(js, 'r'))
            img_path = file['imagePath']
            img_path = osp.join(
                self.img_dir,
                img_path.split('\\')[-1]
                if '\\' in img_path else img_path.split('/')[-1])
            tmp['image_file'] = img_path

            for points in file['shapes']:
                point += points['points'][0]
            tmp['keypoints'] = point
            tmp['point_num'] = len(file['shapes'])
            self.ann_ls.append(tmp)

    def parse_txt(self) -> None:
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
