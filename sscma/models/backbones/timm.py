# Copyright (c) OpenMMLab. All right reserved.
# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from sscma.structures import DataSample
from sscma.datasets import ClsDataPreprocessor



def print_timm_feature_info(feature_info):
    """Print feature_info of timm backbone to help development and debug.

    Args:
        feature_info (list[dict] | timm.models.features.FeatureInfo | None):
            feature_info of timm backbone.
    """
    logger = MMLogger.get_current_instance()
    if feature_info is None:
        logger.warning('This backbone does not have feature_info')
    elif isinstance(feature_info, list):
        for feat_idx, each_info in enumerate(feature_info):
            logger.info(f'backbone feature_info[{feat_idx}]: {each_info}')
    else:
        try:
            logger.info(f'backbone out_indices: {feature_info.out_indices}')
            logger.info(f'backbone out_channels: {feature_info.channels()}')
            logger.info(f'backbone out_strides: {feature_info.reduction()}')
        except AttributeError:
            logger.warning('Unexpected format of backbone feature_info')



class TimmBackbone(BaseModel):
    """Image classifiers for pytorch-image-models (timm) model.

        This class accepts all positional and keyword arguments of the function
        `timm.models.create_model <https://timm.fast.ai/create_model>`_ and use
        it to create a model from pytorch-image-models.

        It can load checkpoints of timm directly, and the saved checkpoints also
        can be directly load by timm.

        Please confirm that you have installed ``timm`` if you want to use it.
    mo
        Args:
            *args: All positional arguments of the function
                `timm.models.create_model`.
            loss (dict): Config of classification loss. Defaults to
                ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
            train_cfg (dict, optional): The training setting. The acceptable
                fields are:

                - augments (List[dict]): The batch augmentation methods to use.
                  More details can be found in :mod:`mmpretrain.model.utils.augment`.

                Defaults to None.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Defaults to False.
            data_preprocessor (dict, optional): The config for preprocessing input
                data. If None or no specified type, it will use
                "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
                more details. Defaults to None.
            init_cfg (dict, optional): the config to control the initialization.
                Defaults to None.
            **kwargs: Other keyword arguments of the function
                `timm.models.create_model`.

        Examples:
            >>> from sscma.models.backbone.timm import TimmBackbone
            >>> model=Timm(model_name='resnet50', loss=dict(type='CrossEntropyLoss'), pretrained=True)
            >>> import torch
            >>> inputs = torch.rand(1, 3, 224, 224)
            >>> out = model(inputs)
            >>> print(out.shape)
            torch.Size([1, 1000])
    """  # noqa: E501

    def __init__(
        self,
        model_name,
        features_only=False,
        pretrained=False,
        checkpoint_path='',
        out_indices=3,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        train_cfg: Optional[dict] = None,
        with_cp: bool = False,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
        **kwargs,
    ):
        if data_preprocessor is None:
            data_preprocessor = {}
        if not features_only:
            #for classification task
            if isinstance(data_preprocessor, dict):
                data_preprocessor.setdefault("type", ClsDataPreprocessor)

            elif not isinstance(data_preprocessor, nn.Module):
                raise TypeError(
                    "data_preprocessor should be a `dict` or "
                    f"`nn.Module` instance, but got "
                    f"{type(data_preprocessor)}"
                )

            if train_cfg is not None and "augments" in train_cfg:
                # Set batch augmentations by `train_cfg`
                data_preprocessor["batch_augments"] = train_cfg


            super().__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        else:
            #for feature extraction task
            super().__init__(init_cfg=init_cfg)

        self.features_only = features_only

        self.model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            # output_stride=8,
            out_indices= out_indices if features_only else None,
            checkpoint_path=checkpoint_path,
            **kwargs)

        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
            self.loss_module = loss


        self.with_cp = with_cp
        if self.with_cp:
            self.model.set_grad_checkpointing()

        # reset classifier
        if hasattr(self.model, 'reset_classifier'):
            self.model.reset_classifier(0, '')

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

        feature_info = getattr(self.model, 'feature_info', None)
        print_timm_feature_info(feature_info)

        self._register_state_dict_hook(self._remove_state_dict_prefix)
        self._register_load_state_dict_pre_hook(self._add_state_dict_prefix)

    @property
    def with_neck(self) -> bool:
        """Whether the classifier has a neck."""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_head(self) -> bool:
        """Whether the classifier has a head."""
        return hasattr(self, "head") and self.head is not None

    def forward(self, inputs, data_samples=None, mode="tensor"):
        if mode == "tensor":
            if self.features_only:
                features = self.model(inputs)
                if isinstance(features, (list, tuple)):
                    features = tuple(features)
                else:
                    features = (features, )
                return features
            else:
                return self.model(inputs)
        elif mode == "loss":
            return self.loss(inputs, data_samples)
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor):
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(inputs)
        else:
            raise NotImplementedError(
                f"The model {type(self.model)} doesn't support extract "
                "feature because it don't have `forward_features` method."
            )

    def loss(self, inputs: torch.Tensor, data_samples: List[DataSample], **kwargs):
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments of the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self.model(inputs)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(
        self, cls_score: torch.Tensor, data_samples: List[DataSample], **kwargs
    ):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if "gt_score" in data_samples[0]:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_score for i in data_samples])
        else:
            target = torch.cat([i.gt_label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(cls_score, target, **kwargs)
        losses["loss"] = loss

        return losses

    def predict(
        self, inputs: torch.Tensor, data_samples: Optional[List[DataSample]] = None
    ):
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.

        Returns:
            List[DataSample]: The prediction results.
        """
        # The part can be traced by torch.fx
        cls_score = self(inputs)

        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score, data_samples=None):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        if data_samples is not None:
            for data_sample, score, label in zip(
                data_samples, pred_scores, pred_labels
            ):
                data_sample.set_pred_score(score).set_pred_label(label)
        else:
            data_samples = []
            for score, label in zip(pred_scores, pred_labels):
                data_samples.append(
                    DataSample().set_pred_score(score).set_pred_label(label)
                )

        return data_samples

    @staticmethod
    def _remove_state_dict_prefix(module, state_dict, prefix, local_metadata):
        for k in list(state_dict.keys()):
            new_key = re.sub(f"^{prefix}model.", prefix, k)
            # Only delete the key that different from its new_key.
            if new_key != k:
                # Modify the `state_dict` directly to avoid invalid changes
                # when recursively calling the `state_dict` function in
                # `torch.nn.module`.
                state_dict[new_key] = state_dict[k]
                del state_dict[k]
        return state_dict

    @staticmethod
    def _add_state_dict_prefix(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        new_prefix = prefix + "model."
        for k in list(state_dict.keys()):
            new_key = re.sub(f"^{prefix}", new_prefix, k)
            # Only delete the key that different from its new_key.
            if new_key != k:
                state_dict[new_key] = state_dict[k]
                del state_dict[k]