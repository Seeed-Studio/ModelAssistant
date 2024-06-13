# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmdet.models.detectors import BaseDetector, SemiBaseDetector
from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import Tensor

from sscma.models.semi import BasePseudoLabelCreator
from sscma.registry import MODELS

from ..utils import samplelist_boxtype2tensor

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class BaseSsod(SemiBaseDetector):
    teacher: BaseDetector
    student: BaseDetector

    def __init__(
        self,
        detector: ConfigType,
        pseudo_label_cfg: ConfigType,
        teacher_loss_weight: int,
        semi_train_cfg: OptConfigType = None,
        semi_test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(detector, semi_train_cfg, semi_test_cfg, data_preprocessor, init_cfg)
        self.pseudo_label_creator: BasePseudoLabelCreator = MODELS.build(pseudo_label_cfg)

        self.teacher_loss_weight = teacher_loss_weight

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
            elif 'sup' in data['inputs'] and not any([True for i in data['inputs']['sup'] if isinstance(i, Tensor)]):
                data['inputs'].pop('sup')
                data['inputs'].pop('sup')
                data['data_samples'].pop('sup')
                data['data_samples'].pop('sup')

            data = self.data_preprocessor(data, True)

            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        # if 'unsup_student' in data['inputs'] and not any(
        #     [True for i in data['inputs']['unsup_student'] if isinstance(i, torch.Tensor)]
        # ):
        #     data['inputs'].pop('unsup_student')
        #     data['inputs'].pop('unsup_teacher')
        #     data['data_samples'].pop('unsup_teacher')
        #     data['data_samples'].pop('unsup_student')
        # elif 'sup' in data['inputs'] and not any([True for i in data['inputs']['sup'] if isinstance(i, Tensor)]):
        #     data['inputs'].pop('sup')
        #     data['data_samples'].pop('sup')

        # print(data['inputs'][0].shape)
        data = self.data_preprocessor(data, False)
        # print(data['inputs'][0].shape)
        return self._run_forward(data, mode='predict')  # type: ignore

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
            # Semi-supervised training process
            with torch.no_grad():
                teacher_pred = self.teacher.predict(
                    multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_student']
                )
            # Generate pseudo labels
            self.pseudo_label_creator.generate_pseudo_labels_online(
                teacher_pred,
                copy.copy(multi_batch_data_samples['unsup_student']),
            )

            losses = dict()
            # Supervision part
            losses.update(**self.loss_by_gt_instances(multi_batch_inputs['sup'], multi_batch_data_samples['sup']))
            # Semi-supervised part
            losses.update(
                **self.loss_by_pseudo_instances(
                    multi_batch_inputs['unsup_student'], multi_batch_data_samples['unsup_student']
                )
            )
        else:
            # Supervise training process
            losses = self.loss_by_gt_instances(multi_batch_inputs['sup'], multi_batch_data_samples['sup'])
        return losses

    def loss_by_pseudo_instances(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, batch_info: Optional[dict] = None
    ) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """
        losses = self.student.loss(batch_inputs, batch_data_samples)
        pseudo_instances_num = sum(
            [len(data_samples.gt_instances) for data_samples in batch_data_samples]
            + [len(data_samples.ignored_instances) for data_samples in batch_data_samples]
        )
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.0) if pseudo_instances_num > 0 else 0.0

        return rename_loss_dict('unsup_', reweight_loss_dict(losses, unsup_weight))


class BaseDetector(BaseModel, metaclass=ABCMeta):
    def __init__(self, data_preprocessor: OptConfigType = None, init_cfg: OptMultiConfig = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self) -> bool:
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head"""
        return (hasattr(self, 'roi_head') and self.roi_head.with_bbox) or (
            hasattr(self, 'bbox_head') and self.bbox_head is not None
        )

    @property
    def with_mask(self) -> bool:
        """bool: whether the detector has a mask head"""
        return (hasattr(self, 'roi_head') and self.roi_head.with_mask) or (
            hasattr(self, 'mask_head') and self.mask_head is not None
        )

    def forward(self, inputs: torch.Tensor, data_samples: OptSampleList = None, mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' 'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, Tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pass

    def add_pred_to_datasample(self, data_samples: SampleList, results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`."""
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples
