# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union, List, Dict, Tuple
import torch

from mmengine.registry import MODELS
from mmengine.model import BaseModel
from sscma.utils.typing_utils import OptConfigType, OptMultiConfig
from sscma.structures import DetDataSample, OptSampleList
from sscma.utils.misc import samplelist_boxtype2tensor

ForwardResults = Union[
    Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor
]


@MODELS.register_module()
class SwiftYOLOQuantModel(BaseModel):
    """SwiftYOLO Quantized Model for QAT training and inference.
    
    This is a template implementation for Swift YOLO QAT support.
    Adapt this implementation based on your specific Swift YOLO model.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
       bbox_head (torch.nn.Module): The detection head module.
       tinynn_model (torch.nn.Module, optional): The quantized backbone model.
    """

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        tinynn_model: torch.nn.Module = None,
        bbox_head: torch.nn.Module = None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self._model = tinynn_model
        self.bbox_head = MODELS.build(bbox_head)

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: OptSampleList = None,
        mode: str = "predict",
    ) -> ForwardResults:
        """The unified entry for a forward process in both training and test.
        
        The method should accept three modes: "tensor", "predict" and "loss":
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.
        """
        if mode == "predict":
            return self._predict(inputs, data_samples)
        elif mode == "loss":
            return self._loss(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". QuantModel only supports predict and loss modes'
            )

    def _predict(self, inputs: torch.Tensor, data_samples: OptSampleList) -> List[DetDataSample]:
        """Predict results from the quantized model."""
        # Forward through the quantized backbone
        data = self._model(inputs)
        
        # Extract metadata for prediction
        batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        
        # Use the detection head to process features and generate predictions
        results = self.bbox_head.predict_by_feat(
            *data, batch_img_metas=batch_img_metas
        )
        
        # Assign predictions to data samples
        for result, data_sample in zip(results, data_samples):
            data_sample.pred_instances = result

        # Convert box types to tensor format for consistency
        samplelist_boxtype2tensor(data_samples)
        return data_samples

    def _loss(self, inputs: torch.Tensor, batch_data_samples: OptSampleList) -> Dict[str, torch.Tensor]:
        """Compute loss for the quantized model during training."""
        # Forward through the quantized backbone
        data = self._model(inputs)
        
        # Prepare loss inputs based on your Swift YOLO head implementation
        # This example follows the RTMDet pattern - adapt based on your head's loss_by_feat signature
        loss_inputs = data + (
            batch_data_samples["bboxes_labels"],  # Ground truth labels
            batch_data_samples["img_metas"],      # Image metadata
        )
        
        # Compute losses using the detection head
        losses = self.bbox_head.loss_by_feat(*loss_inputs)
        return losses
    
    def set_model(self, model: torch.nn.Module):
        """Set the quantized model.
        
        This method is called by the quantization training script
        to set the quantized backbone model.
        
        Args:
            model (torch.nn.Module): The quantized backbone model.
        """
        self._model = model
        
    def extract_feat(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        """Extract features from the quantized backbone.
        
        This method can be used for debugging or feature visualization.
        
        Args:
            inputs (torch.Tensor): Input images.
            
        Returns:
            Tuple[torch.Tensor]: Extracted features.
        """
        if self._model is None:
            raise RuntimeError("Quantized model is not set. Call set_model() first.")
        return self._model(inputs)