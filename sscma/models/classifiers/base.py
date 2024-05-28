# Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) OpenMMLab.
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from sscma.registry import MODELS
from sscma.structures import ClsDataSample
from mmengine.model import BaseModel


class BaseClassifier(BaseModel, metaclass=ABCMeta):
    """Base class for classifiers.

    Args:
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None, it will use "BaseDataPreprocessor" as type, see
            :class:`mmengine.model.BaseDataPreprocessor` for more details.
            Defaults to None.

    Attributes:
        init_cfg (dict): Initialization config dict.
        data_preprocessor (:obj:`mmengine.model.BaseDataPreprocessor`): An
            extra data pre-processing module, which processes data from
            dataloader to the format accepted by :meth:`forward`.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'sscma.ClsDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super(BaseClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)
        
        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    @property
    def with_neck(self) -> bool:
        """Whether the classifier has a neck."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        """Whether the classifier has a head."""
        return hasattr(self, 'head') and self.head is not None

    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[ClsDataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
          tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`BaseDataElement`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...)
                in general.
            data_samples (List[BaseDataElement], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmengine.BaseDataElement`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs: torch.Tensor, stage='neck'):
        """Extract features from the input tensor with shape (N, C, ...).

        The sub-classes are recommended to implement this method to extract
        features from backbone and neck.

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
        """
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(inputs)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        assert self.with_head and hasattr(self.head, 'pre_logits'), \
            "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def extract_feats(self, multi_inputs: Sequence[torch.Tensor],
                      **kwargs) -> list:
        """Extract features from a sequence of input tensor.

        Args:
            multi_inputs (Sequence[torch.Tensor]): A sequence of input
                tensor. It can be used in augmented inference.
            **kwargs: Other keyword arguments accepted by :meth:`extract_feat`.

        Returns:
            list: Features of every input tensor.
        """
        assert isinstance(multi_inputs, Sequence), \
            '`extract_feats` is used for a sequence of inputs tensor. If you '\
            'want to extract on single inputs tensor, use `extract_feat`.'
        return [self.extract_feat(inputs, **kwargs) for inputs in multi_inputs]
    
    def loss(self, inputs: torch.Tensor,
             data_samples: List[ClsDataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[ClsDataSample]] = None,
                **kwargs) -> List[ClsDataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[ClsDataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples, **kwargs)
