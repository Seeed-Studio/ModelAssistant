import os.path as osp
from collections import OrderedDict

import cv2
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from mmdet.datasets.coco import CocoDataset

from .utils.download import check_file
