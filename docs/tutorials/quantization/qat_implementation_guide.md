# QAT Support for Custom Models in SSCMA

This document provides a comprehensive guide on how to add Quantization Aware Training (QAT) support for custom models in SSCMA, specifically addressing the Swift YOLO QAT implementation request.

## Background

The SSCMA framework currently supports QAT for RTMDet models. This guide explains how to extend that support to other custom models like Swift YOLO, following the same architectural patterns.

## Understanding QAT in SSCMA

### Core Components

The QAT implementation in SSCMA consists of several key components:

1. **Quantization Training Script** (`tools/quantization.py`)
   - Main entry point for QAT training
   - Handles model quantization using TinyNeuralNetwork
   - Manages training loop and model export

2. **Quantizer Switch Hook** (`sscma/engine/hooks/quantizer_switch_hook.py`)
   - Manages QAT training phases
   - Controls when to freeze quantizer parameters and batch norm statistics

3. **Quantized Model Wrapper** (e.g., `sscma/quantizer/models/rtmdet_quantizer.py`)
   - Wraps the quantized model for training and inference
   - Handles forward pass and loss computation

4. **QAT Configuration Files** (e.g., `configs/rtmdet/rtmdet_nano_8xb32_300e_coco_relu_q.py`)
   - Defines training parameters specific to QAT
   - Configures the quantized model wrapper

### QAT Training Process

The QAT training process follows these steps:

1. **Model Loading**: Load the pre-trained model
2. **Cross-Layer Equalization**: Apply optimization techniques
3. **Quantization Setup**: Initialize QAT quantizer with fake quantization
4. **Model Wrapping**: Wrap the quantized model with the quantizer wrapper
5. **QAT Training**: Train the model with quantization-aware gradients
6. **Model Export**: Convert to actual quantized model and export to TFLite

## Implementing QAT for Swift YOLO

### Prerequisites

Before implementing QAT for Swift YOLO, ensure you have:

1. A working Swift YOLO implementation
2. Pre-trained Swift YOLO model weights
3. Understanding of your Swift YOLO model architecture
4. Dataset properly configured for training

### Implementation Steps

#### Step 1: Create the Quantized Model Wrapper

The quantized model wrapper is the core component that adapts your model for QAT training.

**Key Points:**
- Must inherit from `BaseModel`
- Should handle both `predict` and `loss` modes
- Needs to work with the quantized backbone from TinyNeuralNetwork
- Must match the input/output format of your original model

**Template Location:** `sscma/quantizer/models/swift_yolo_quantizer.py`

#### Step 2: Register the Quantized Model

Update the quantizer models registry to include your new quantized model wrapper.

**Files to Modify:**
- `sscma/quantizer/models/__init__.py`

#### Step 3: Create QAT Configuration

Create a configuration file that defines:
- Training parameters optimized for QAT
- Data preprocessing pipeline
- Quantized model wrapper configuration
- Learning rate schedule
- QAT-specific hooks

**Template Location:** `configs/swift_yolo/swift_yolo_qat_template.py`

#### Step 4: Test and Validate

1. Test the quantized model wrapper with dummy inputs
2. Run QAT training with a small dataset
3. Compare quantized model performance with the original
4. Validate TFLite export functionality

### Adaptation Guidelines

When adapting the template for your specific Swift YOLO implementation:

#### Model Architecture Considerations

1. **Backbone Compatibility**: Ensure your backbone architecture is compatible with TinyNeuralNetwork quantization
2. **Head Interface**: The detection head must provide `predict_by_feat` and `loss_by_feat` methods
3. **Feature Flow**: Verify that feature tensors flow correctly between quantized backbone and head

#### Loss Computation

The `_loss` method in your quantized model wrapper must:
- Accept the same input format as your original model
- Use the same loss computation logic as your detection head
- Return losses in the expected format

#### Data Pipeline

QAT training may require specific data preprocessing:
- Input normalization should match your model's expectations
- Augmentations should be compatible with quantization training
- Batch size may need adjustment for QAT stability

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: Module import failures when loading the quantized model wrapper.

**Solution**: 
- Ensure all dependencies are properly imported
- Check that Swift YOLO components are registered in the MODELS registry
- Verify module paths and naming consistency

#### 2. Shape Mismatches

**Problem**: Tensor shape mismatches between quantized backbone and detection head.

**Solution**:
- Debug the forward pass with small test inputs
- Compare shapes between original and quantized models
- Ensure head expects the same input format

#### 3. Loss Computation Errors

**Problem**: Errors during loss computation in QAT training.

**Solution**:
- Verify that the loss inputs format matches your head's expectations
- Check that all required metadata is properly passed
- Compare loss computation between original and quantized models

#### 4. Training Instability

**Problem**: QAT training shows unstable loss or poor convergence.

**Solution**:
- Reduce learning rate for QAT training
- Adjust the quantizer freezing schedule
- Use gradient clipping if necessary
- Ensure proper batch norm handling

### Performance Optimization

#### Training Speed

1. **Batch Size**: Larger batch sizes can improve QAT training stability
2. **Precision**: Use mixed precision training where compatible
3. **Data Loading**: Optimize data loading for QAT training pipeline

#### Model Quality

1. **Calibration Data**: Use representative calibration data for quantization setup
2. **Training Schedule**: Fine-tune the learning rate schedule for your model
3. **Hyperparameters**: Adjust quantization parameters for your specific use case

### Testing Your Implementation

#### Unit Testing

Create tests to verify:
- Quantized model wrapper initialization
- Forward pass with dummy inputs
- Loss computation with sample data
- Model export functionality

#### Integration Testing

Test the complete QAT pipeline:
- End-to-end training with small dataset
- Model export to TFLite format
- Inference accuracy comparison
- Performance benchmarking

### Example Usage

Once implemented, use your Swift YOLO QAT as follows:

```bash
# QAT Training
python tools/quantization.py \
    configs/swift_yolo/swift_yolo_qat_config.py \
    path/to/swift_yolo_pretrained.pth \
    --work-dir work_dirs/swift_yolo_qat \
    --cfg-options epochs=5

# Testing
python tools/quantization.py \
    configs/swift_yolo/swift_yolo_qat_config.py \
    work_dirs/swift_yolo_qat/epoch_5.pth \
    --test \
    --work-dir work_dirs/swift_yolo_qat_test
```

### Further Resources

- **TinyNeuralNetwork Documentation**: For detailed quantization parameters
- **SSCMA RTMDet QAT Implementation**: Reference implementation in the codebase
- **PyTorch Quantization Guide**: For understanding quantization concepts
- **SSCMA Model Development Guide**: For general model development in SSCMA

## Conclusion

Implementing QAT support for Swift YOLO follows the same patterns as the existing RTMDet implementation. The key is to create an appropriate quantized model wrapper that handles the specific requirements of your model architecture while maintaining compatibility with the SSCMA QAT training pipeline.

For questions or issues during implementation, refer to the existing RTMDet QAT code as a reference and ensure that all components follow the same architectural patterns.