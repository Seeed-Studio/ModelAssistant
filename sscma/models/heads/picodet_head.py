# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS, TASK_UTILS
from sscma.models.task_modules.prior_generators.point_generator import (
    MlvlPointGenerator,
)
from sscma.models.task_modules.samplers import PseudoSampler
from sscma.structures.bbox import (
    bbox2distance,
    distance2bbox,
)
from sscma.structures.bbox import bbox_overlaps
from sscma.utils.dist_utils import reduce_mean

from sscma.utils.misc import filter_scores_and_topk, select_single_mlvl, images_to_levels
from sscma.utils.misc import multi_apply
from .anchor_free_head import AnchorFreeHead
from ..cnn import ConvModule, DepthwiseSeparableConvModule


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x




class PicoDetHead(AnchorFreeHead):
    """PicoDetHead head used in `PicoDet <https://arxiv.org/abs/2111.00902>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 96
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolutions.
            Default: True
        kernel_size (int): Kernel size of convolution layer. Default: 5
        share_cls_reg (bool): If ture,reg and cls branch will share weights.
            Default: True
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: BN
        act_cfg (dict): Config dict for activation layer. Default: HSwish.
        use_vfl (bool): Whether use vfl loss for classification. Defualt: True
        loss_cls (dict): Config of classification loss.
        loss_dfl (dict): Config of localization loss.
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in DFL setting. Default: 7.
        sync_num_pos (bool): If true, synchronize the number of positive
            examples across GPUs. Default: True
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels=96,
            stacked_convs=2,
            strides=[8, 16, 32, 64],
            use_depthwise=True,
            kernel_size=5,
            share_cls_reg=True,
            conv_cfg=None,
            act_cfg=dict(type='HSwish'),
            norm_cfg=dict(type='BN', requires_grad=True),
            use_vfl=True,
            loss_cls=dict(
                type='VarifocalLoss',
                use_sigmoid=True,
                alpha=0.75,
                gamma=2.0,
                iou_weighted=True,
                loss_weight=1.0),
            loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
            reg_max=7,
            sync_num_pos=True,
            init_cfg=dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal',
                    name='gfl_cls',
                    std=0.01,
                    bias_prob=0.01)),
            **kwargs):
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.reg_max = reg_max
        self.share_cls_reg = share_cls_reg
        self.act_cfg = act_cfg
        self.ConvModule = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        super(PicoDetHead, self).__init__(
            num_classes, in_channels, stacked_convs=stacked_convs, strides=strides,
            feat_channels=feat_channels, loss_cls=loss_cls, norm_cfg=norm_cfg, init_cfg=init_cfg, **kwargs)
        self.sampling = False
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            if self.train_cfg.get("sampler", None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg.sampler, default_args=dict(context=self)
                )
            else:
                self.sampler = PseudoSampler(context=self)
        self.prior_generator = MlvlPointGenerator(strides, offset=0.5)
        self.sync_num_pos = sync_num_pos
        self.integral = Integral(self.reg_max)
        self.loss_dfl = MODELS.build(loss_dfl)
        self.use_vfl = use_vfl
        if use_vfl:
            assert loss_cls['type'] == 'VarifocalLoss', 'when set use_vfl=True, loss_cls type must be VarifocalLoss'
            self.loss_cls = MODELS.build(loss_cls)

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.share_cls_reg
                    else self.cls_out_channels,
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )
        if not self.share_cls_reg:
            self.gfl_reg = nn.ModuleList(
                [
                    nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 1, padding=0)
                    for _ in self.strides
                ]
            )
        else:
            self.gfl_reg = [None] * len(self.strides)

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=(self.kernel_size - 1) // 2,
                    act_cfg=self.act_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    self.ConvModule(
                        chn,
                        self.feat_channels,
                        self.kernel_size,
                        stride=1,
                        padding=(self.kernel_size - 1) // 2,
                        act_cfg=self.act_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None,
                    )
                )

        return cls_convs, reg_convs

    def forward(self, feats):
        return multi_apply(
            self.forward_single,
            feats,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
        )

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg):
        cls_feat = x
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)

        if not self.share_cls_reg:
            reg_feat = x
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
        if self.share_cls_reg:
            feat = gfl_cls(cls_feat)
            cls_score, bbox_pred = torch.split(
                feat, [self.cls_out_channels, 4 * (self.reg_max + 1)], dim=1
            )
        else:
            cls_score = gfl_cls(cls_feat)
            bbox_pred = gfl_reg(reg_feat)

        if torch.onnx.is_in_onnx_export():
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = (
                torch.sigmoid(cls_score)
                .reshape(1, self.num_classes, -1)
                .permute(0, 2, 1)
            )
            bbox_pred = bbox_pred.reshape(1, (self.reg_max + 1) * 4, -1).permute(
                0, 2, 1
            )
        return cls_score, bbox_pred

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_imgs = len(img_metas)
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_scores]

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device,
            with_stride=True) # anchor points

        decode_bbox_preds = []
        center_and_strides = []
        for stride, bbox_pred, center_and_stride in zip(self.strides, bbox_preds, mlvl_priors):
            center_and_stride = center_and_stride.repeat(num_imgs, 1, 1)
            center_and_strides.append(center_and_stride)
            center_in_feature = center_and_stride.reshape(
                (-1, 4))[:, :-2] / stride
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
            pred_corners = self.integral(bbox_pred)
            decode_bbox_pred = distance2bbox(
                center_in_feature, pred_corners).reshape(num_imgs, -1, 4)
            decode_bbox_preds.append(decode_bbox_pred * stride)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(
                [num_imgs, -1, self.cls_out_channels])
            for cls_pred in cls_scores
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1) # N x #points x #cls_out_channels
        flatten_bboxes = torch.cat(decode_bbox_preds, dim=1)
        flatten_center_and_strides = torch.cat(center_and_strides, dim=1)

        cls_reg_targets = self.get_targets(
            flatten_cls_preds,
            flatten_bboxes,
            flatten_center_and_strides,
            gt_bboxes,
            gt_labels,
            num_level_anchors)

        labels_list, label_weights_list, bbox_targets_list, \
            center_and_strides_list, num_total_pos = cls_reg_targets
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                labels_list[0].new_tensor(num_total_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_total_pos
        loss_bbox_list, loss_dfl_list, loss_cls_list, avg_factor = [], [], [], []
        for cls_score, bbox_pred, center_and_strides, labels, label_weights, bbox_targets, stride in zip(
                cls_scores, bbox_preds, center_and_strides_list, labels_list,
                label_weights_list, bbox_targets_list, self.strides):
            center_and_strides = center_and_strides.reshape(-1, 4)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(
                [-1, self.cls_out_channels]
            )
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
                -1, 4 * (self.reg_max + 1)
            )
            bbox_targets = bbox_targets.reshape(-1, 4)
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)

            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            bg_class_ind = self.num_classes

            pos_inds = torch.nonzero((labels >= 0)  & (labels < bg_class_ind)).squeeze(1)

            if self.use_vfl:
                score = label_weights.new_zeros(cls_score.shape)
            else:
                score = label_weights.new_zeros(labels.shape)

            if num_total_pos > 0:
                pos_bbox_targets = bbox_targets[pos_inds]
                pos_bbox_pred = bbox_pred[pos_inds]
                pos_centers = center_and_strides[:, :-2][pos_inds] / stride

                weight_targets = cls_score.detach().sigmoid()
                weight_targets = weight_targets.max(dim=1)[0][pos_inds]
                pos_bbox_pred_corners = self.integral(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_centers,
                                                     pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride

                pos_ious = bbox_overlaps(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets.detach(),
                    is_aligned=True).clamp(min=1e-6)
                if self.use_vfl:
                    pos_labels = labels[pos_inds]
                    score[pos_inds, pos_labels] = pos_ious.clone().detach()
                else:
                    score[pos_inds] = pos_ious.clone().detach()
                pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
                target_corners = bbox2distance(pos_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape(-1)
                # regression loss
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets.detach(),
                    weight=weight_targets,
                    avg_factor=1.0)
                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                    avg_factor=4.0
                )
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = bbox_pred.new_tensor(0)
            # cls (qfl) loss
            if self.use_vfl:
                loss_cls = self.loss_cls(
                    cls_score,
                    score,
                    avg_factor=num_pos_avg_per_gpu)
            else:
                loss_cls = self.loss_cls(
                    cls_score, (labels, score),
                    weight=label_weights,
                    avg_factor=num_pos_avg_per_gpu
                )
            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_cls_list.append(loss_cls)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
        losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
        losses_cls = sum(loss_cls_list)
        losses = dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

        return losses

    @torch.no_grad()
    def _get_target_single(self,
                           cls_preds,
                           priors,
                           decoded_bboxes,
                           gt_bboxes,
                           gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        num_bboxes = decoded_bboxes.shape[0]
        # # No target
        if num_gts == 0:
            labels = priors.new_full((num_bboxes, ),
                                     self.num_classes,
                                     dtype=torch.long)
            label_weights = priors.new_zeros(num_bboxes, dtype=torch.float)
            bbox_targets = torch.zeros_like(decoded_bboxes)
            return (0, labels, label_weights, bbox_targets)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid(),
            priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        num_pos_per_img = pos_inds.size(0)
        bbox_targets = torch.zeros_like(decoded_bboxes)
        bbox_weights = torch.zeros_like(decoded_bboxes)
        labels = priors.new_full((num_bboxes, ),
                                 self.num_classes,
                                 dtype=torch.long)
        label_weights = priors.new_zeros(num_bboxes, dtype=torch.float)

        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (num_pos_per_img, labels, label_weights, bbox_targets)

    def loss_single(self, cls_score, bbox_pred, center_and_strides, labels, label_weights,
                    bbox_targets, stride, num_total_samples):
        """Compute loss of a single scale level.
        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W).
            bbox_pred (Tensor): Box
                level with shape (N, num_points * 4, H, W).
            center_and_strides (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4)
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_points).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_points)
            bbox_targets (Tensor): BBox regression targets of each points
            weight shape (N, num_total_points, 4).
            stride (int): Stride for each scale level
            num_total_samples (int) If sampling, num total samples equal to
                the number of total points; Otherwise, it is the number of
                positive points.
        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """

        center_and_strides = center_and_strides.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            [-1, self.cls_out_channels]
        )
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, 4 * (self.reg_max + 1)
        )
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        # pos_inds = ((labels >= 0)
        #             & (labels < bg_class_ind)).nonzero().squeeze(1)
        pos_inds = torch.nonzero((labels >= 0)  & (labels < bg_class_ind)).squeeze(1)

        if self.use_vfl:
            score = label_weights.new_zeros(cls_score.shape)
        else:
            score = label_weights.new_zeros(labels.shape)

        if num_total_samples > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_centers = center_and_strides[:, :-2][pos_inds] / stride

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_centers,
                                                 pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride

            pos_ious = bbox_overlaps(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets.detach(),
                is_aligned=True).clamp(min=1e-6)
            if self.use_vfl:
                pos_labels = labels[pos_inds]
                score[pos_inds, pos_labels] = pos_ious.clone().detach()
            else:
                score[pos_inds] = pos_ious.clone().detach()
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(pos_centers,
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)
            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets.detach(),
                weight=weight_targets,
                avg_factor=1.0)
            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0
            )
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)
        # cls (qfl) loss
        if self.use_vfl:
            loss_cls = self.loss_cls(
                cls_score,
                score,
                avg_factor=num_total_samples)
        else:
            loss_cls = self.loss_cls(
                cls_score, (labels, score),
                weight=label_weights,
                avg_factor=num_total_samples
            )
            print(loss_cls, loss_bbox, loss_dfl, weight_targets.sum())


        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   score_factors=None,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].device,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def get_targets(self,
                    flatten_cls_preds,
                    flatten_bboxes,
                    flatten_center_and_strides,
                    gt_bboxes_list,
                    gt_labels_list,
                    num_level_anchors,
                    ):
        """Compute regression and classification targets for points in
        multiple images.

        Args:
            flatten_cls_pred (Tensor) Flattened classification predictions of images.
            flatten_bboxes (Tensor) Flattened bbox predictions of images.
            flatten_center_and_strides (Tensor) Flattened anchor center and strides predictions of images
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            num_level_anchors (list[int]): Number points of each scale.
        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - center_and_strides_list (list[Tensor]): Center and stride of each level
                - num_total_pos (int): Number of positive samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        pos_num_l, label_l, label_weight_l, bbox_target_l = [], [], [], []

        for flatten_cls_pred, flatten_center_and_stride, flatten_bbox, gt_bbox, gt_label \
                in zip(flatten_cls_preds.detach(), flatten_center_and_strides.detach(), \
                       flatten_bboxes.detach(), gt_bboxes_list, gt_labels_list):
            pos_num, label, label_weight, bbox_target = self._get_target_single(
                flatten_cls_pred, flatten_center_and_stride, flatten_bbox,
                gt_bbox, gt_label)
            pos_num_l.append(pos_num)
            label_l.append(label)
            label_weight_l.append(label_weight)
            bbox_target_l.append(bbox_target)

        center_and_strides_list = images_to_levels([flatten_cs for flatten_cs in flatten_center_and_strides],
                                                   num_level_anchors)
        labels_list = images_to_levels(label_l, num_level_anchors)
        label_weights_list = images_to_levels(label_weight_l,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(bbox_target_l,
                                             num_level_anchors)

        num_total_pos = sum([max(pos_num, 1) for pos_num in pos_num_l])

        return (labels_list, label_weights_list, bbox_targets_list, center_and_strides_list, num_total_pos)

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = distance2bbox(
                priors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._bbox_post_process(
            mlvl_scores,
            mlvl_labels,
            mlvl_bboxes,
            img_meta['scale_factor'],
            cfg,
            rescale=rescale,
            with_nms=with_nms)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Hack some keys of the model state dict so that can load checkpoints
        of previous version."""
        version = local_metadata.get('version', None)
        version = 'ppdet2mmdet'
        local_metadata['version'] = 'ppdet2mmdet'
        if version is None:
            # the key is different in early versions
            # for example, 'fcos_cls' become 'conv_cls' now
            bbox_head_keys = [
                k for k in state_dict.keys() if k.startswith(prefix)
            ]
            ori_predictor_keys = []
            new_predictor_keys = []
            # e.g. 'fcos_cls' or 'fcos_reg'
            for key in bbox_head_keys:
                ori_predictor_keys.append(key)
                key = key.split('.')
                conv_name = None
                if key[1].endswith('cls'):
                    conv_name = 'conv_cls'
                elif key[1].endswith('reg'):
                    conv_name = 'conv_reg'
                elif key[1].endswith('centerness'):
                    conv_name = 'conv_centerness'
                else:
                    assert NotImplementedError
                if conv_name is not None:
                    key[1] = conv_name
                    new_predictor_keys.append('.'.join(key))
                else:
                    ori_predictor_keys.pop(-1)
            for i in range(len(new_predictor_keys)):
                state_dict[new_predictor_keys[i]] = state_dict.pop(
                    ori_predictor_keys[i])
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def onnx_export(self,
                    cls_scores,
                    bbox_preds,
                    score_factors=None,
                    img_metas=None,
                    with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            score_factors (list[Tensor]): score_factors for each s
                cale level with shape (N, num_points * 1, H, W).
                Default: None.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc. Default: None.
            with_nms (bool): Whether apply nms to the bboxes. Default: True.

        Returns:
            tuple[Tensor, Tensor] | list[tuple]: When `with_nms` is True,
            it is tuple[Tensor, Tensor], first tensor bboxes with shape
            [N, num_det, 5], 5 arrange as (x1, y1, x2, y2, score)
            and second element is class labels of shape [N, num_det].
            When `with_nms` is False, first tensor is bboxes with
            shape [N, num_det, 4], second tensor is raw score has
            shape  [N, num_det, num_classes].
        """
        return cls_scores, bbox_preds