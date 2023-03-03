# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models.builder import CLASSIFIERS, build_backbone, build_head, build_neck
from mmcls.models.heads import MultiLabelClsHead
from mmcls.models.utils.augment import Augments
from mmcls.models.classifiers.base import BaseClassifier


@CLASSIFIERS.register_module()
class AccelerometerClassifier(BaseClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(AccelerometerClassifier, self).__init__(init_cfg)

        if pretrained is not None:
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.augments = None
        if train_cfg is not None:
            augments_cfg = train_cfg.get('augments', None)
            if augments_cfg is not None:
                self.augments = Augments(augments_cfg)

    def forward_dummy(self, img):
        
        img = self.extract_feat(img , stage='backbone')
        
        if self.with_head:
            return self.head.forward_dummy(img)

        return img
    
    def extract_feat(self, img, stage='neck'):

        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        img = self.backbone(img)

        if stage == 'backbone':
            return img

        if self.with_neck:
            img = self.neck(img)
        if stage == 'neck':
            return img

        if self.with_head and hasattr(self.head, 'pre_logits'):
            img = self.head.pre_logits(img)
        return img

    def forward_train(self, img, gt_label, **kwargs):

        if self.augments is not None:
            img, gt_label = self.augments(img, gt_label)

        img = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(img, gt_label)

        losses.update(loss)

        return losses

    def simple_test(self, img, img_metas=None, **kwargs):

        img = self.extract_feat(img)

        if isinstance(self.head, MultiLabelClsHead):
            assert 'softmaimg' not in kwargs, (
                'Please use `sigmoid` instead of `softmaimg` '
                'in multi-label tasks.')
        res = self.head.simple_test(img, **kwargs)

        return res

    def forward(self, img, flag=False, return_loss=True, **kwargs):
        
        if (flag):
            return self.forward_dummy(img)

        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)
