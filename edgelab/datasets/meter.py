import os
import math
import copy
import json
import os.path as osp
from abc import ABCMeta
from typing import Optional, Sequence

import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from edgelab.registry import DATASETS

from .utils.download import check_file
from .pipelines.composition import AlbCompose


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
    """
    The meter data set class, this class is mainly for the data set of 
    the pointer table, the data set is marked in a format similar to the 
    key point detection
    
    Args:
        data_root: The root path of the dataset
        index_file: The path of the annotation file or the folder path
            of the annotation file
        img_dir: The folder path of the image data, which needs to be 
            the image file name that can be found in the corresponding 
            annotation file
        pipeline: The option to do data enhancement on image data, which
            needs to be in list format
        format: format of keypoints. Should be 'xy', 'yx', 'xya', 'xys', 'xyas', 'xysa'
    
    """
    CLASSES = ('meter')

    def __init__(self,
                 data_root: str,
                 index_file: str,
                 img_dir: Optional[str] = None,
                 pipeline: Optional[Sequence[dict]] = None,
                 format: str = 'xy'):
        super(MeterData, self).__init__()
        self.metainfo = dict()

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
        elif index_file.endswith('txt'):
            self.parse_txt(index_file)
        elif index_file.endswith('json'):
            self.parse_json(index_file)
        else:
            raise ValueError(
                'The parameter index_file must be a folder path',
                ' or a file in txt or json format, but the received ',
                f'value is {index_file}')

        self.transforms = AlbCompose(pipeline, keypoint_params=format)
        self.totensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item: int) -> dict:
        ann = copy.deepcopy(self.ann_ls[item])
        img_file = ann['image_file']
        self.img = cv2.imread(img_file)
        ann['init_size'] = self.img.shape
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

        return {'inputs': img, 'data_samples': ann}

    def __len__(self) -> int:
        return len(self.ann_ls)

    def parse_jsons(self, index_dir: str) -> None:
        """
        When the annotation file is in json format, parse the corresponding annotation file
        """
        file_ls = os.listdir(index_dir)
        file_ls = [osp.join(index_dir, i) for i in file_ls]

        self.ann_ls = []
        for js in file_ls:
            tmp = {}
            point = []
            file = json.load(open(js, 'r'))
            img_path = file['imagePath']
            
            sep='\\' if '\\' in img_path else '/'
            img_path = osp.join(img_path.split(sep)[-1])
            
            tmp['image_file'] = img_path

            for points in file['shapes']:
                point += points['points'][0]
            tmp['keypoints'] = point
            tmp['point_num'] = len(file['shapes'])
            self.ann_ls.append(tmp)

    def parse_txt(self, index_file: str) -> None:
        """
        When the comment file is in txt format, parse the corresponding comment file
        """
        with open(index_file, 'r') as f:
            self.lines = f.readlines()

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

    def parse_json(self, json_path: str) -> None:
        pass  #todo