# Swift YOLO QAT Implementation Guide

This guide explains how to implement Quantization Aware Training (QAT) support for Swift YOLO models, based on the existing RTMDet QAT implementation.

## Overview

QAT (Quantization Aware Training) allows models to be trained with quantization in mind, resulting in better performance when deployed on edge devices with limited precision. SSCMA currently supports QAT for RTMDet models, and this guide shows how to extend that support to Swift YOLO.

## Prerequisites

1. Swift YOLO model implementation (available on branch 2.0.0)
2. Understanding of the SSCMA model architecture
3. Basic knowledge of PyTorch quantization

## Implementation Steps

### Step 1: Create a Swift YOLO Quantized Model Wrapper

Following the pattern of `RtmdetQuantModel`, you need to create a quantized model wrapper for Swift YOLO. This wrapper handles the quantized forward pass and loss computation.

**File:** `sscma/quantizer/models/swift_yolo_quantizer.py`

```python
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

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
       bbox_head (torch.nn.Module): The detection head module.
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
        """Forward pass for quantized Swift YOLO model."""
        if mode == "predict":
            data = self._model(inputs)
            batch_img_metas = [data_samples.metainfo for data_samples in data_samples]
            results = self.bbox_head.predict_by_feat(
                *data, batch_img_metas=batch_img_metas
            )
            
            for result, data_sample in zip(results, data_samples):
                data_sample.pred_instances = result

            samplelist_boxtype2tensor(data_samples)
            return data_samples
        elif mode == "loss":
            return self._loss(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". QuantModel only supports predict and loss modes'
            )

    def _loss(self, inputs: torch.Tensor, batch_data_samples: OptSampleList):
        """Compute loss for quantized model."""
        data = self._model(inputs)
        # Adapt this based on your Swift YOLO head's loss computation
        loss_inputs = data + (
            batch_data_samples["bboxes_labels"],
            batch_data_samples["img_metas"],
        )
        losses = self.bbox_head.loss_by_feat(*loss_inputs)
        return losses
    
    def set_model(self, model):
        """Set the quantized model."""
        self._model = model
```

### Step 2: Register the New Quantized Model

Update the quantizer models `__init__.py` file:

**File:** `sscma/quantizer/models/__init__.py`

```python
from .rtmdet_quantizer import RtmdetQuantModel
from .pfld_quantizer import PFLDQuantModel
from .fomo_quantizer import FomoQuantizer
from .anomaly_quantizer import AnomalyQuantModel
from .swift_yolo_quantizer import SwiftYOLOQuantModel  # Add this line

__all__ = [
    "RtmdetQuantModel", 
    "PFLDQuantModel", 
    "FomoQuantizer", 
    "AnomalyQuantModel",
    "SwiftYOLOQuantModel"  # Add this line
]
```

### Step 3: Create QAT Configuration File

Create a QAT configuration file for Swift YOLO based on your base Swift YOLO config:

**File:** `configs/swift_yolo/swift_yolo_qat_example.py`

```python
# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

# Import your base Swift YOLO configuration
with read_base():
    from .swift_yolo_base import *  # Replace with actual base config

from sscma.datasets.transforms.loading import LoadImageFromFile
from sscma.datasets.transforms.processing import RandomResize
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, ConstantLR

from sscma.datasets.transforms.formatting import PackDetInputs
from sscma.datasets.transforms.loading import LoadAnnotations
from sscma.datasets.transforms.transforms import (
    Pad,
    RandomCrop,
    RandomFlip,
    Resize,
    HSVRandomAug,
)
from sscma.engine.schedulers import QuadraticWarmupLR
from sscma.engine.hooks import QuantizerSwitchHook
from sscma.quantizer import SwiftYOLOQuantModel

# QAT specific settings
imgsz = (640, 640)  # Adjust based on your model requirements
dump_config = False

max_epochs = 5
num_last_epochs = 2
base_lr = 0.00002

# QAT training pipeline
train_pipeline = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend="pillow",
        backend_args=None,
    ),
    dict(type=LoadAnnotations, imdecode_backend="pillow", with_bbox=True),
    dict(type=HSVRandomAug),
    dict(
        type=RandomResize,
        scale=(imgsz[0] * 2, imgsz[1] * 2),
        ratio_range=(0.5, 1.5),
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs),
]

# Update model configuration for QAT
model.bbox_head.update(train_cfg=model.train_cfg)
model.bbox_head.update(test_cfg=model.test_cfg)

# Configure the quantized model wrapper
quantizer_config = dict(
    type=SwiftYOLOQuantModel,
    bbox_head=model.bbox_head,
    data_preprocessor=model.data_preprocessor,
)

# Update training configuration
train_dataloader.update(
    dict(batch_size=32, num_workers=16, dataset=dict(pipeline=train_pipeline))
)

train_cfg.update(
    dict(
        type=EpochBasedTrainLoop,
        max_epochs=max_epochs,
        val_interval=1,
        val_begin=1,
        dynamic_intervals=None,
    )
)

# Optimizer configuration for QAT
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type=QuadraticWarmupLR,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=base_lr * 0.05,
        begin=1,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type=ConstantLR,
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    ),
]

# QAT specific hooks
custom_hooks = [
    dict(
        type=QuantizerSwitchHook,
        freeze_quantizer_epoch=max_epochs // 3,
        freeze_bn_epoch=max_epochs // 3 * 2,
    ),
]
```

### Step 4: Usage Instructions

Once you have implemented the above components, you can use QAT training for Swift YOLO as follows:

```bash
# Train with QAT
python tools/quantization.py configs/swift_yolo/swift_yolo_qat_example.py \
    path/to/your/swift_yolo_pretrained.pth \
    --work-dir work_dirs/swift_yolo_qat \
    --cfg-options epochs=5

# Test the quantized model
python tools/quantization.py configs/swift_yolo/swift_yolo_qat_example.py \
    work_dirs/swift_yolo_qat/epoch_5.pth \
    --test \
    --work-dir work_dirs/swift_yolo_qat_test
```

## Key Points to Remember

1. **Model Architecture Compatibility**: Ensure your Swift YOLO model architecture is compatible with the quantization process.

2. **Head Configuration**: The `bbox_head` configuration in the quantizer must match your Swift YOLO head implementation.

3. **Loss Computation**: The `_loss` method in the quantized model wrapper should match how your Swift YOLO head computes losses.

4. **Input/Output Format**: Ensure the quantized model wrapper handles the same input/output format as your original Swift YOLO model.

5. **Testing**: Always test the quantized model performance compared to the original model to ensure acceptable accuracy retention.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure Swift YOLO is properly imported and registered in the MODELS registry.

2. **Shape Mismatches**: Verify that the tensor shapes between the quantized backbone and head are compatible.

3. **Loss Computation Errors**: Check that the loss computation in the quantized model matches the original implementation.

### Debug Tips

1. Compare the forward pass outputs between the original and quantized models using small test inputs.

2. Use the existing RTMDet QAT implementation as a reference for debugging.

3. Enable verbose logging during training to monitor the quantization process.

## Next Steps

1. Implement the Swift YOLO quantized model wrapper based on your specific Swift YOLO implementation.

2. Create and test the QAT configuration file.

3. Run QAT training and evaluate the results.

4. Fine-tune hyperparameters for optimal quantized model performance.

For more details about the quantization process, refer to the existing RTMDet QAT implementation and the TinyNeuralNetwork documentation.