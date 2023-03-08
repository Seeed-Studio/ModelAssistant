import torch
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck


@DETECTORS.register_module()
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
        self.backbone = build_backbone(backbone)
        self.bbox_head = build_head(head)
        if neck:
            self.neck = build_neck(neck)

    def forward(self, img, target, flag=False, return_loss=True, **kwargs):
        if flag:
            return torch.softmax(self.forward_dummy(img))
        else:
            if return_loss:
                # extract image feature
                x = self.extract_feat(img)
                result = self.bbox_head(x)
                return self.bbox_head.loss(result, target)
            else:
                return self.forward_test(img, label=target)

    def forward_test(self, imgs, **kwargs):

        x = self.extract_feat(imgs)

        result = self.bbox_head(x)
        return result.permute(0, 2, 3, 1), self.bbox_head.build_target(
            result.permute(0, 2, 3, 1), kwargs['label'])
        # return self.bbox_head.post_handle(result,kwargs['label'])

    def train_step(self, data, optimizer):
        losses = self(**data)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss,
                       log_vars=log_vars,
                       num_samples=len(data['img']))

        return outputs
