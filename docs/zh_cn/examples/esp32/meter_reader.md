# 使用 ESP32 实现表计读数

本教程将基于 EdgeLab 示范使用 ESP32 实现表计读数的开发流程。

::: tip

在开始前，我们建议您先阅读 [ESP32 - 部署教程](./deploy.md)。

:::

## 准备工作

请参考 [ESP32 - 部署教程 - 先决条件](./deploy.md#%E5%85%88%E5%86%B3%E6%9D%A1%E4%BB%B6)。

## 训练模型

表计读数功能基于 PFLD 模型实现，在这一步您需要一个后缀为 `.pth` 的 PFLD 模型权重，您有两种方法获取该模型权重:

- 在我们的 [Model Zoo](https://github.com/Seeed-Studio/edgelab-model-zoo) 下载预训练好的模型。

- 参考[模型训练 - PFLD 模型](../../tutorials/training/pfld.md)，基于 PyTorch 和 EdgeLab 自行训练 PFLD 模型得到模型权重。

## 导出模型

由于训练得到的模型并不适合直接在边缘计算设备上运行，我们首先需要将其导出为后缀是 `.tflite` 的 TFLite 格式，您有两种方法获取导出的模型 (包含模型权重):

- 在我们的 [Model Zoo](https://github.com/Seeed-Studio/edgelab-model-zoo) 下载导出为 TFLite 格式的模型。

- 参考[模型导出 - PyTorch 转 TFLite](../../tutorials/export/pytorch_2_tflite.md)，自行将 PFLD 模型从 PyTorch 格式转换为 TFLite 格式。

## 转换模型

在完成[导出模型](#%E5%AF%BC%E5%87%BA%E6%A8%A1%E5%9E%8B)后，我们还需要进一步处理，将其转换为嵌入式设备支持的格式。

- 进入 `examples/esp32` 目录 (在 EdgeLab 项目根目录运行):

  ```sh
  cd examples/esp32
  ```

- 转换 TFLite 模型为二进制 C 文件

  ```sh
  python3 tools/tflite2c.py --input <TFLITE_MODEL_PATH> --name fomo --output_dir components/modules/model --classes='("unmask", "mask")'
  ```

::: tip

您需要将 `<TFLITE_MODEL_PATH>` 替换为在[导出模型](#%E5%AF%BC%E5%87%BA%E6%A8%A1%E5%9E%8B)步骤取得的 TFLite 模型的路径，转换生成的 C 文件将默认输出到 `EdgeLab/example/esp32` 目录中的 `components/modules/model` 目录。

:::

## 部署模型

这是完成表计读数的最后一步，也是最重要的一步，在这一步您需要编译并刷写固件到 ESP32 MCU。请参考 [ESP32 - 部署教程 - 编译和部署](./deploy.md#%E7%BC%96%E8%AF%91%E5%92%8C%E9%83%A8%E7%BD%B2)完成模型的部署。

## 运行示例

![PFLD Meter Reader](/static/esp32/images/pfld_meter.gif)
