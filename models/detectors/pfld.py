import cv2
import mmcv
import os.path as osp

from mmpose.models.builder import build_loss
from mmpose.models.detectors.base import BasePose
from mmpose.models.builder import MESH_MODELS, build_backbone

from models.utils.computer_acc import pose_acc


@MESH_MODELS.register_module()
class PFLD(BasePose):

    def __init__(self, backbone, loss_cfg, pretrained=None):
        super(PFLD, self).__init__()
        self.backbone = build_backbone(backbone)
        self.computer_loss = build_loss(loss_cfg)
        self.pretrained = pretrained

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()

    def forward(self, img, keypoints=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, keypoints, **kwargs)
        else:
            return self.forward_test(img, keypoints, **kwargs)

    def forward_train(self, img, keypoints, **kwargs):
        x = self.backbone(img)
        acc = pose_acc(x.cpu().detach().numpy(),
                       keypoints.cpu().detach().numpy(), kwargs['hw'])
        return {'loss': self.computer_loss(x, keypoints), 'Acc': acc}

    def forward_test(self, img, keypoints, **kwargs):
        x = self.backbone(img)
        result = {}
        if keypoints is not None:
            loss = self.computer_loss(x, keypoints)
            acc = pose_acc(x.cpu().detach().numpy(), keypoints, kwargs['hw'])
            result['loss'] = loss
            result['Acc'] = acc
        result.update({'result': x, **kwargs})
        return result

    def forward_dummy(self, img, **kwargs):
        x = self.backbone(img)
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

        # img=imshow_keypoints(img, [keypoints.cpu()])
        for point in keypoints:
            if not isinstance(point, (float, int)):
                img = cv2.circle(img, (int(point[0]), int(point[1])), 2,
                                 (255, 0, 0), -1)
        if show:
            cv2.imshow(win_name, img)
            cv2.waitKey(500)

        if save_path:
            img_name = osp.basename(img_file)
            cv2.imwrite(osp.join(save_path, img_name), img)
