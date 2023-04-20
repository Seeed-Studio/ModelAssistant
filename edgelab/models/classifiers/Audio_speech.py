from edgelab.registry import MODELS, LOSSES
from mmcls.models.classifiers.base import BaseClassifier
from mmengine.logging import MessageHub
import torch


@MODELS.register_module("Audio_classify", force=True)
class Audio_classify(BaseClassifier):
    """
    https://arxiv.org/abs/2204.11479
    END-TO-END AUDIO STRIKES BACK: BOOSTING
    AUGMENTATIONS TOWARDS AN EFFICIENT AUDIO
    CLASSIFICATION NETWORK
    """

    def __init__(self,
                 backbone,
                 n_cls,
                 loss=dict(),
                 multilabel=False,
                 data_preprocessor=None,
                 head=None,
                 loss_cls=None,
                 pretrained=None):
        super(BaseClassifier, self).__init__()
        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(head)
        self.cls_loss = MODELS.build(loss_cls)
        if data_preprocessor is not None:
            self.data_preprocessor = MODELS.build(data_preprocessor)
        self.pretrained = pretrained
        self.sm = torch.nn.Softmax(1)
        self.n_cls = n_cls
        self.mutilabel = multilabel
        self._loss = LOSSES.build(loss)

    def forward(self, img, mode='loss', **kwargs):
        if mode == 'loss':
            return self.loss(img, **kwargs)
        elif mode == 'predict':
            return self.predict(img, **kwargs)
        elif mode == 'tensor':
            return self.predict(img, **kwargs)

    def loss(self, img, **kwargs):
        features = self.backbone(img)
        result = self.cls_head(features)

        if MessageHub.get_current_instance().get_info('ismixed'):
            target = MessageHub.get_current_instance().get_info('target')
            loss = MessageHub.get_current_instance().get_info(
                'audio_loss').mix_loss(result,
                                       target,
                                       self.n_cls,
                                       pred_one_hot=self.mutilabel)
        else:
            loss = self._loss(result, kwargs['labels'])

        return {'loss': loss}

    def predict(self, img, **kwargs):
        features = self.backbone(img)
        result = self.sm(self.cls_head(features))
        # return [{'pred_label':{"score":result},"gt_label":{"label":kwargs['labels']}}]
        return [{
            'pred_label': {
                "label": torch.max(result, dim=1)[1]
            },
            "gt_label": {
                "label": kwargs['labels']
            }
        }]
