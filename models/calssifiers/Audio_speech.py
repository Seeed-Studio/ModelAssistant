from mmcls.models.builder import CLASSIFIERS, build_backbone, build_head, build_loss
from mmcls.models.classifiers.base import BaseClassifier
import torch


@CLASSIFIERS.register_module("Audio_classify", force=True)
class Audio_classify(BaseClassifier):
    def __init__(self, backbone, head=None, loss_cls=None, pretrained=None):
        super(BaseClassifier, self).__init__()
        self.backbone = build_backbone(backbone)
        self.cls_head = build_head(head)
        self.cls_loss = build_loss(loss_cls)
        self.pretrained = pretrained

    def forward_train(self, img, **kwargs):
        features = self.backbone(img)
        result = self.cls_head(features)
        return {'loss': self.cls_loss(result, kwargs['labels']),
                'acc': (kwargs['labels'] == torch.max(result, dim=1)[1]).float().mean()}

    def extract_feat(self, imgs, stage=None):
        pass

    def simple_test(self, img, **kwargs):
        features = self.backbone(img)
        result = self.cls_head(features)
        return [{'loss': self.cls_loss(result, kwargs['labels']),
                 'acc': (kwargs['labels'] == torch.max(result, dim=1)[1]).float().mean()}]
