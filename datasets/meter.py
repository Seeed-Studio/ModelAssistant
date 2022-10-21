import math
import numpy as np
import cv2
import albumentations as A
import torch

from mmpose.datasets.builder import DATASETS
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def calc_angle(x1, y1, x2, y2):
    x = (x1 - x2)
    y = (y1 - y2)
    z = math.sqrt(x * x + y * y)
    try:
        angle = math.acos((z ** 2 + 1 - (x - 1) ** 2 - y ** 2) / (2 * z * 1)) / math.pi * 180
    except:
        angle = 0

    if y < 0:
        angle = 360 - angle

    return angle


@DATASETS.register_module()
class MeterData(Dataset):
    CLASSES = ('meter')

    def __init__(self, index_file, test_mode=None, transform=None):
        super(MeterData, self).__init__()

        self.train = transform
        self.test_trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(size=(112, 112), interpolation=InterpolationMode.NEAREST)])
        self.transforms = A.Compose([
            A.ColorJitter(brightness=0.3, p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.5),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Rotate(border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_NEAREST,),
            # A.ChannelShuffle(),
            # A.SafeRotate(border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_NEAREST, p=0.5),
            # A.CoarseDropout(max_holes=4,max_height=32,max_width=32,p=0.5),
            # A.RandomCrop(height=160,width=160,p=0.5),
            A.Affine(translate_percent=[0.05, 0.1], mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST, p=0.6),
            # A.CoarseDropout(max_holes=4,max_height=32,max_width=32,p=0.5),
        ], keypoint_params=A.KeypointParams(format='xy'))

        self.point_num = 1

        with open(index_file, 'r') as f:
            self.lines = f.readlines()
        self.flag = np.zeros(self.__len__(), dtype=np.uint8)

    def __getitem__(self, item):
        self.line = self.lines[item].strip().split()
        img_file = './datasets/' + self.line[0].replace('\\', '/')          #todo
        img_file = str(img_file)
        self.img = cv2.imread(img_file)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.landmark = np.asarray(self.line[1:5], dtype=np.float32)

        self.landmark = np.asarray(self.line[1:], dtype=np.float32)
        point = []
        point.append(self.landmark[0])
        point.append(self.landmark[1])
        landmark = []
        for i in range(self.point_num):
            landmark.append([point[i * 2], point[i * 2 + 1]])

        if self.train:
            while True:
                result = self.transforms(image=self.img, keypoints=landmark)
                if len(result['keypoints']) < self.point_num:
                    continue
                else:
                    break
            img, label = self.test_trans(result['image']), np.asarray(result['keypoints']).flatten() / 240
        else:
            img, label = self.test_trans(self.img), np.asarray(landmark).flatten() / 240

        return {'img': img, 'img_metas': label}

    def __len__(self):
        return len(self.lines)

    def evaluate(self, results, **kwargs):
        return {'loss': torch.mean(torch.tensor([i['loss'] for i in results])).cpu().item()}
