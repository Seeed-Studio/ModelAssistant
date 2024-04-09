# Copyright (c) Seeed Tech Ltd. All rights reserved.
from abc import abstractmethod

import torch.nn as nn
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList

from sscma.registry import MODELS


@MODELS.register_module()
class BasePseudoLabelCreator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def generate_pseudo_labels_online_(
        self, teach_pred: InstanceList, student_samples: SampleList, teacher_samples: OptSampleList = None
    ):
        """
        Online generation of pseudo-labels through correlation strategies
        Args:
            teacher_pred (InstanceList): Prediction results obtained after the unlabeled image
                passes through the teacher model
            student_samples (SampleList): Relevant information that needs to be input to the
                student model image sampling,This parameter needs to be passed through shallow
                copy. After the pseudo label is generated, the generated pseudo label will be
                passed to this parameter.
            teacher_samples (OptSampleList): Information related to the sampling entered into
                the teacher model image
        Returns: None
        """
