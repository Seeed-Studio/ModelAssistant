# 使用 Grove - Vision AI 实现数字标记

本教程将基于 [SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 示范使用 Grove - Vision AI 模块实现数字读数的开发流程。

::: tip

在开始前，我们建议您先阅读 [Grove - 部署教程](./deploy.md)。

:::

## 准备工作

请参考 [Grove - 部署教程 - 先决条件](./deploy.md#%E5%85%88%E5%86%B3%E6%9D%A1%E4%BB%B6)。

## 训练模型

表计读数功能基于 YOLOv5 模型实现，在这一步您需要一个后缀为 `.pth` 的 YOLOv5 模型权重，您有两种方法获取该模型权重:

- 在我们的 [Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo) 下载预训练好的模型。

- 参考[模型训练 - YOLOv5 模型](../../tutorials/training/yolo.md)，基于 PyTorch 和 [SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 自行训练 YOLOv5 模型得到模型权重。

## 导出模型

由于训练得到的模型并不适合直接在边缘计算设备上运行，我们首先需要将其导出为后缀是 `.tflite` 的 TFLite 格式，您有两种方法获取导出的模型 (包含模型权重):

- 在我们的 [Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo) 下载导出为 TFLite 格式的模型。

- 参考[模型导出 - PyTorch 转 TFLite](../../tutorials/export/pytorch_2_tflite.md)，自行将 YOLOv5 模型从 PyTorch 格式转换为 TFLite 格式。

## 部署模型

这是完成表计读数的最后一步，也是最重要的一步，在这一步您需要编译并刷写固件到 Grove - Vision AI 模块。请参考 [Grove - 部署教程 - 编译和部署](./deploy.md#%E7%BC%96%E8%AF%91%E5%92%8C%E9%83%A8%E7%BD%B2)完成模型的部署。

## 运行示例

在完成 [Grove - 部署教程 - 编译和部署 - 部署例程](./deploy.md#%E9%83%A8%E7%BD%B2%E4%BE%8B%E7%A8%8B)步骤后，打开 [Grove Vision AI 控制台](https://files.seeedstudio.com/grove_ai_vision/index.html)

以上步骤在控制台均有图形化的提示，最后，您可以看到实时表盘读数结果如下图所示。

![YOLOv5 Digital Meter Reader](/static/grove/images/digital_meter.gif)
