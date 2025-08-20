# QAT Documentation and Templates for SSCMA

This directory contains comprehensive documentation and templates for implementing Quantization Aware Training (QAT) support for custom models in SSCMA.

## Files Overview

### Documentation
- **`qat_implementation_guide.md`**: Comprehensive guide explaining the QAT architecture in SSCMA and general implementation principles
- **`swift_yolo_qat_guide.md`**: Specific guide for implementing QAT support for Swift YOLO models

### Templates
- **`sscma/quantizer/models/swift_yolo_quantizer.py`**: Template quantized model wrapper for Swift YOLO
- **`configs/swift_yolo/swift_yolo_qat_template.py`**: Template QAT configuration file

## Background

This documentation was created in response to Issue #297, where a user requested QAT support for Swift YOLO models. While QAT is supported for RTMDet on the main branch, Swift YOLO is on the 2.0.0 branch and lacks QAT support.

## Usage

1. **Read the Documentation**: Start with `qat_implementation_guide.md` to understand the overall QAT architecture
2. **Follow the Swift YOLO Guide**: Use `swift_yolo_qat_guide.md` for step-by-step implementation
3. **Adapt the Templates**: Modify the provided templates based on your specific Swift YOLO implementation
4. **Test and Validate**: Follow the testing guidelines to ensure proper implementation

## Key Implementation Points

- The quantized model wrapper must handle both `predict` and `loss` modes
- Loss computation in the wrapper must match your original model's logic
- Configuration files need QAT-specific training parameters
- The existing QAT training pipeline (`tools/quantization.py`) can be used without modification

## Implementation Checklist

- [ ] Create quantized model wrapper based on template
- [ ] Update quantizer models `__init__.py` to register new wrapper
- [ ] Create QAT configuration file based on template
- [ ] Test with small dataset
- [ ] Validate TFLite export functionality
- [ ] Compare quantized vs original model performance

## Support

For implementation questions:
1. Refer to the existing RTMDet QAT implementation as a reference
2. Follow the patterns established in `sscma/quantizer/models/rtmdet_quantizer.py`
3. Use the same training pipeline and hooks as RTMDet QAT

The templates and documentation provide a complete foundation for implementing QAT support for any custom model in SSCMA, not just Swift YOLO.