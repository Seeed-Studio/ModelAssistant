# Copyright (c) Seeed Tech Ltd. All rights reserved.
from typing import Dict, Union

import torch
from mmdet.models.detectors import BaseDetector, SemiBaseDetector
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.optim import OptimWrapper
from torch import Tensor

from sscma.models.semi.labelmatch import LabelMatch
from sscma.registry import LOSSES, MODELS


@MODELS.register_module()
class EfficientTeacher(SemiBaseDetector):
    teacher: BaseDetector
    student: BaseDetector

    def __init__(
        self,
        detector: ConfigType,
        domain_loss_cfg: ConfigType,
        target_loss_cfg: ConfigType,
        pseudo_label_cfg: ConfigType,
        teacher_loss_weight: int,
        da_loss_weight: int = 0,
        online_pseudo: bool = True,
        semi_train_cfg: OptConfigType = None,
        semi_test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(detector, semi_train_cfg, semi_test_cfg, data_preprocessor, init_cfg)
        self.pseudo_label_creator: LabelMatch = MODELS.build(pseudo_label_cfg)
        self.domain_loss = LOSSES.build(domain_loss_cfg)
        self.target_loss = LOSSES.build(target_loss_cfg)
        self.teacher_loss_weight = teacher_loss_weight
        self.da_loss_weight = da_loss_weight
        self.online_pseudo = online_pseudo

    def train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper) -> Dict[str, Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            if 'unsup_student' in data['inputs'] and not any(
                [True for i in data['inputs']['unsup_student'] if isinstance(i, torch.Tensor)]
            ):
                data['inputs'].pop('unsup_student')
                data['inputs'].pop('unsup_teacher')
                data['data_samples'].pop('unsup_teacher')
                data['data_samples'].pop('unsup_student')

            data = self.data_preprocessor(data, True)

            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def loss(self, multi_batch_inputs: Dict[str, Tensor], multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """
        if multi_batch_inputs.get('unsup_teacher', None) is not None:
            return super().loss(multi_batch_inputs, multi_batch_data_samples)
        else:
            # supervised training process
            losses = self.loss_by_gt_instances(multi_batch_inputs['sup'], multi_batch_data_samples['sup'])
        return losses

    def computer_domain_loss(self, feature: Tensor) -> Tensor:
        return self.domain_loss(feature)

    def computer_target_loss(self, feature: Tensor) -> Tensor:
        return self.target_loss(feature)
