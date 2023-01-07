import torch
import torch.nn as nn
from mmdet.core import bbox2result
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_loss, build_neck


@DETECTORS.register_module()
class FastestDet(SingleStageDetector):

    def __init__(
        self,
        backbone,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=dict(nms_pre=1000,
                      min_bbox_size=0,
                      score_thr=0.05,
                      conf_thr=0.005,
                      nms=dict(type='nms', iou_threshold=0.45),
                      max_per_img=100),
        pretrained=None,
        init_cfg=None,
    ):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained, init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.bbox_head = build_head(bbox_head)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def extract_feat(self, img):
        s1, s2, s3 = self.backbone(img)

        if hasattr(self, 'neck'):
            s3 = self.upsample(s3)
            s1 = self.avg_pool(s1)
            s = torch.concat((s1, s2, s3), dim=1)
            x = self.neck(s)
            return x
        return s1, s2, s3

    def forward(self, img, img_metas, flag=False, return_loss=True, **kwargs):
        if flag:
            return self.forward_dummy(img)
        else:
            if return_loss:
                return self.forward_train(img, img_metas, **kwargs)
            else:
                return self.forward_test(img, img_metas, **kwargs)

    def forward_test(self, img, img_metas, **kwargs):
        for imgs, img_meta in zip(img, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(imgs.size()[-2:])

        x = self.extract_feat(img[0])

        result = self.bbox_head(x)
        if 'fomo' in kwargs.keys():
            return self.bbox_head.post_handle(result)

        results_list = self.bbox_head.handle_preds(
            result, result.device, img_metas[0][0]['ori_shape'][:2])
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results
