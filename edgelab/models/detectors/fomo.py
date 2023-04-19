from mmdet.models.detectors.single_stage import SingleStageDetector
from edgelab.registry import MODELS


@MODELS.register_module()
class Fomo(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(backbone, neck, head, train_cfg, test_cfg, pretrained,
                         init_cfg)

    def forward(self, img, target, mode='loss'):
        if mode == 'loss':
            return self.loss(img, target)
        elif mode == 'predict':
            return self.predict(img, label=target)
        elif mode == 'tensor':
            return self.predict(img, label=target)
        else:
            raise ValueError(
                f'params mode recive a not exception params:{mode}')

    def loss(self, img, target):
        x = self.extract_feat(img)
        return self.bbox_head.loss(x, target)

    def predict(self, imgs, **kwargs):
        x = self.extract_feat(imgs)
        return self.bbox_head.predict(x, **kwargs)
