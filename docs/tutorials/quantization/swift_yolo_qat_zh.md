# Swift YOLO 量化感知训练 (QAT) 支持

本文档针对 Issue #297 提供了为 Swift YOLO 添加 QAT 支持的完整解决方案。

## 问题背景

用户询问如何为 Swift YOLO 添加 QAT（量化感知训练）支持。虽然主分支上的 RTMDet 已经支持 QAT，但 Swift YOLO 在 2.0.0 分支上，缺少 QAT 支持。

## 解决方案

我们基于现有的 RTMDet QAT 实现，为用户提供了完整的 Swift YOLO QAT 支持模板和文档。

### 提供的文件和模板

#### 1. 文档
- **`docs/tutorials/quantization/qat_implementation_guide.md`** - QAT 实现的综合指南
- **`docs/tutorials/quantization/swift_yolo_qat_guide.md`** - Swift YOLO QAT 的具体实现指南
- **`docs/tutorials/quantization/README.md`** - 文档和模板的总览

#### 2. 代码模板
- **`sscma/quantizer/models/swift_yolo_quantizer.py`** - Swift YOLO 量化模型包装器模板
- **`configs/swift_yolo/swift_yolo_qat_template.py`** - QAT 配置文件模板
- **`examples/swift_yolo_qat_example.py`** - 使用示例脚本

#### 3. 代码修改
- **`sscma/quantizer/models/__init__.py`** - 注册新的量化模型

## 实现步骤

### 第 1 步：创建量化模型包装器
使用提供的模板 `swift_yolo_quantizer.py` 来创建您的 Swift YOLO 量化模型包装器。这个包装器需要：
- 处理量化后的前向传播
- 实现损失计算
- 与 TinyNeuralNetwork 兼容

### 第 2 步：注册量化模型
模板已经更新了 `__init__.py` 文件来注册新的量化模型。

### 第 3 步：创建 QAT 配置文件
基于您的 Swift YOLO 基础配置，使用提供的模板创建 QAT 配置文件。

### 第 4 步：训练和测试
使用现有的量化训练脚本：

```bash
# QAT 训练
python tools/quantization.py configs/swift_yolo/your_qat_config.py \
    path/to/swift_yolo_pretrained.pth \
    --work-dir work_dirs/swift_yolo_qat

# 测试
python tools/quantization.py configs/swift_yolo/your_qat_config.py \
    work_dirs/swift_yolo_qat/epoch_5.pth \
    --test
```

## 核心要点

1. **模型架构兼容性**：确保您的 Swift YOLO 架构与量化过程兼容
2. **头部接口**：检测头部必须提供 `predict_by_feat` 和 `loss_by_feat` 方法
3. **损失计算**：量化模型包装器中的损失计算必须与原始模型逻辑匹配
4. **数据流**：验证特征张量在量化主干网络和头部之间正确流动

## 适配建议

根据您的具体 Swift YOLO 实现调整模板：

1. **更新模型配置**：在配置文件中定义您的 Swift YOLO 模型结构
2. **适配损失函数**：根据您的检测头部实现调整 `_loss` 方法
3. **调整数据管道**：确保数据预处理与您的模型期望匹配
4. **测试验证**：比较量化前后的模型性能

## 框架相同

如维护者所说，框架是相同的。这个实现遵循与 RTMDet QAT 完全相同的模式：

- 使用相同的量化训练脚本 (`tools/quantization.py`)
- 使用相同的量化开关钩子 (`QuantizerSwitchHook`)
- 遵循相同的量化模型包装器模式
- 使用相同的 TinyNeuralNetwork 量化后端

## 支持和参考

- 参考现有的 RTMDet QAT 实现作为基准
- 使用提供的模板作为起点
- 遵循文档中的最佳实践
- 如有问题，可以参考 RTMDet 的实现模式

这个解决方案提供了完整的框架，让您可以轻松地为 Swift YOLO 添加 QAT 支持，而不需要修改核心的量化训练基础设施。