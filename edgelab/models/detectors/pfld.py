import cv2
import mmcv
import os.path as osp
import torch
import numpy as np

from mmpose.models.pose_estimators.base import BasePoseEstimator
from mmengine.registry import MODELS

from edgelab.models.utils.computer_acc import pose_acc


@MODELS.register_module()
class PFLD(BasePoseEstimator):

    def __init__(self, backbone, head, loss_cfg, pretrained=None):
        super(PFLD, self).__init__(backbone,head=head)
        self.backbone = MODELS.build(backbone)
        self.head = MODELS.build(head)
        self.computer_loss = MODELS.build(loss_cfg)
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

    def loss(self, inputs, data_samples) -> dict:
        return super().loss(inputs, data_samples)
    
    def predict(self, inputs, data_samples):
        return super().predict(inputs, data_samples)

    def forward(self,
                img,
                keypoints=None,
                mode='loss',
                **kwargs):
        if mode=='predict':
            print(kwargs)

        if mode=='loss':
            return self.forward_train(img, keypoints, **kwargs)
        elif mode=='predict':
            return self.forward_dummy(img,**kwargs)
        elif mode=='tensor':
            pass
        else:
            raise ValueError(f'params mode recive a not exception params:{mode}')

    def forward_train(self, img, keypoints, **kwargs):
        x = self.backbone(img)
        x = self.head(x)
        # acc = pose_acc(x[0].cpu().detach().numpy(),
        #                keypoints[0], kwargs['hw'])

        return {'loss': self.computer_loss(x, torch.tensor(keypoints,device=torch.device('cuda:0')))}

    def forward_test(self, img, keypoints, **kwargs):
        x = self.backbone(img)
        x = self.head(x)
        result = {}
        if keypoints is not None:
            loss = self.computer_loss(x, keypoints)
            acc = pose_acc(x.cpu().detach().numpy(), 
                           keypoints.cpu().detach().numpy(), kwargs['hw'])
            result['loss'] = loss
            result['Acc'] = acc
        result.update({'result': x, **kwargs})
        return result

    def forward_dummy(self, img, **kwargs):
        x = self.backbone(img)
        x = self.head(x)
        return x

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
