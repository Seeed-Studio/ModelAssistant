import os.path as osp

import cv2
import mmcv
import torch
import numpy as np

from mmpose.models.pose_estimators.base import BasePoseEstimator
from edgelab.registry import MODELS
from mmpose.structures import PoseDataSample
from mmengine.structures.instance_data import InstanceData


@MODELS.register_module()
class PFLD(BasePoseEstimator):

    def __init__(self, backbone, head, pretrained=None):
        super(PFLD, self).__init__(backbone, head=head)
        self.backbone = MODELS.build(backbone)
        self.head = MODELS.build(head)

        self.pretrained = pretrained

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        # self.backbone.init_weights(self.pretrained)
        # if self.with_neck:
        #     self.neck.init_weights()
        # if self.with_keypoint:
        #     self.keypoint_head.init_weights()

    def forward(self, inputs, data_samples, mode='loss'):

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self.predict(inputs, data_samples)
        else:
            raise ValueError(
                f'params mode recive a not exception params:{mode}')

    def loss(self, inputs, data_samples):
        # inputs=torch.stack(inputs)
        x = self.extract_feat(inputs)
        results = dict()
        results.update(
            self.head.loss(
                x,
                torch.tensor(data_samples['keypoints'],
                             device=torch.device('cuda:0')),
                data_samples['hw']))
        return results

    def predict(self, inputs, data_samples):
        # inputs=torch.stack(inputs)
        feat = self.extract_feat(inputs)
        x = self.head.predict(feat)
        res = PoseDataSample(**data_samples)
        res.results = x
        res.pred_instances = InstanceData(
            keypoints=np.array([x.cpu().numpy()]) *
            data_samples['init_size'][1].cpu().numpy())
        return [res]

    def show_result(self,
                    img_file,
                    keypoints,
                    show=False,
                    win_name='img',
                    save_path=None,
                    **kwargs):
        img = mmcv.imread(img_file, channel_order='bgr').copy()
        h, w = img.shape[:-1]
        keypoints[::2] = keypoints[::2] * w
        keypoints[1::2] = keypoints[1::2] * h
        keypoints = keypoints.cpu().numpy()

        for idx, point in enumerate(keypoints[::2]):
            if not isinstance(point, (float, int)):
                img = cv2.circle(img,
                                 (int(point), int(keypoints[idx * 2 + 1])), 2,
                                 (255, 0, 0), -1)
        if show:
            cv2.imshow(win_name, img)
            cv2.waitKey(500)

        if save_path:
            img_name = osp.basename(img_file)
            cv2.imwrite(osp.join(save_path, img_name), img)
