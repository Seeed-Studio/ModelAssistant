from mmpose.models.builder import MESH_MODELS, build_backbone
from mmpose.models.detectors.base import BasePose
from mmpose.models.builder import build_loss


@MESH_MODELS.register_module()
class PFLD(BasePose):
    def __init__(self, backbone, loss_cfg, pretrained=None):
        super(PFLD, self).__init__()
        self.backbone = build_backbone(backbone)
        self.computer_loss = build_loss(loss_cfg)
        self.pretrained = pretrained

    def init_weights(self):
        pass

    def forward_train(self, img, img_metas, **kwargs):
        pass

    def forward_test(self, img, img_metas, **kwargs):
        pass

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        x = self.backbone(img)
        if return_loss:
            return {'loss': self.computer_loss(x, img_metas)}

        return {'loss': self.computer_loss(x, img_metas)}

    def show_result(self, **kwargs):
        pass

    def forward_dummy(self, img_metas, **kwargs):
        x = self.backbone(img_metas)
        return x

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        if pretrained is not None:
            self.pretrained = pretrained
        self.backbone.init_weights(self.pretrained)
        if self.with_neck:
            self.neck.init_weights()
        if self.with_keypoint:
            self.keypoint_head.init_weights()
