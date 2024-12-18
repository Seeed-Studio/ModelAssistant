# 在 Espressif 芯片上部署模型

本示例为 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 包含的模型在 Espreessif 芯片的部署教程，部署工作基于 [ESP-IDF](https://github.com/espressif/esp-idf) 和 [Tensorflow Lite Micro](https://github.com/tensorflow/tflite-micro) 实现。

## 先决条件

### 硬件

- 一台 Linux 或者 macOS 计算机

- 一块带摄像头的 ESP32-S3 开发板 (例如 [Seeed Studio XIAO](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html))

- 一根 USB 数据线

### 安装 ESP-IDF

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 包含的模型在 ESP32 的部署需要 ESP-IDF `5.1.x`，请参考以下教程 [ESP-IDF Get Started Guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)，安装配置工具链和 ESP-IDF。

在成功 ESP-IDF 安装后，请再次确认[配置 IDF 环境变量](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables)是否完成:

- 设置了 `IDF_PATH` 环境变量。

- 确定 `idf.py` 和 Xtensa-ESP32 等工具（例如 `xtensa-esp32-elf-gcc`）都在包含在 `$PATH` 中。

:::tip

我们不建议您在虚拟环境中配置 ESP-IDF，您可以使用以下命令退出虚拟环境 (可多次使用退出嵌套的虚拟环境):

```sh
conda deactivate
```

此外，如果您的 ESP-IDF 未配置在虚拟环境中，任何有关 ESP-IDF 的操作，如 `idf.py` 的调用，都应该在非虚拟环境中进行。

:::

### 获取示例和子模块

**进入 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 项目的根目录**，运行下面的命令来获取示例和子模块。

```sh
git clone https://github.com/Seeed-Studio/sscma-example-esp32 -b 1.0.0  examples/esp32 && \
pushd examples/esp32 && \
git submodule init && \
git submodule update && \
popd
```

:::warning

您需要提前完成 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 的安装与配置。如果您还没有安装 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant), 请参考[SSCMA 安装指南](../../introduction/installation)。

:::

## 准备模型

在开始编译和部署之前，您需要先根据实际应用场景，准备好需要部署的模型。因此，您可能需要经历模型或神经网络的选择、自定义数据集、导出或转换模型等步骤。

为了让您更有条理地理解该过程，我们针对不同的应用场景编写了完整的文档 [SSCMA - 模型训练与导出](../training/overview.md)。

:::warning

在[编译和部署](#%E7%BC%96%E8%AF%91%E5%92%8C%E9%83%A8%E7%BD%B2)前，您需要提前准备好相应的模型。

:::

## 编译和部署

### 编译例程

1. 进入 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 项目的根目录，运行以下命令进入示例目录 `examples`:

```sh
cd examples/<examples>
```

2. 设置 `IDF_TARGET` 为 `esp32s3`:

```sh
idf.py set-target esp32s3
```

3. 编译例程:

```sh
idf.py build
```

### 部署例程

1. 将 ESP32 MCU 连接到计算机，确定 ESP32 的串口路径。在 Linux 下，您可以使用以下命令来检查当前可用的串口 (在 Linux 上新接入的 ESP32 设备，串口路径一般为 `/dev/ttyUSB0`):

```sh
lsusb -t && \
ls /dev | grep tty
```

2. 刷写固件 (请将 `<TARGET_SERIAL_PORT>` 替换为 ESP32 的串口路径):

```sh
idf.py --port <TARGET_SERIAL_PORT> flash
```

3. 监控串口输出并等待 MCU 重启 (请将 `<TARGET_SERIAL_PORT>` 替换为 ESP32 的串口路径):

```sh
idf.py --port <TARGET_SERIAL_PORT> monitor
```

:::tip

刷写固件和监控串口的两条命令可以合并使用:

```sh
idf.py --port <TARGET_SERIAL_PORT> flash monitor
```

使用 `Ctrl+]` 来退出监控串口输出界面。

:::

### 性能简介

通过在不同的芯片上测量，对 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 相关模型的性能总结如下表所示。

| Target | Model | Dataset | Input Resolution | Peak RAM | Inferencing  Time | F1 Score | Link |
|--|--|--|--|--|--|--|--|
| ESP32-S3 | Meter | [Custom Meter](https://files.seeedstudio.com/sscma/datasets/meter.zip) | 112x112 (RGB) | 320KB | 380ms | 97% | [pfld_meter_int8.tflite](https://github.com/Seeed-Studio/ModelAssistant/releases/tag/model_zoo) |
| ESP32-S3 | Fomo | [COCO MASK](https://files.seeedstudio.com/sscma/datasets/coco_mask.zip) | 96x96 (GRAY) | 244KB | 150ms | 99.5% | [fomo_mask_int8.tflite](https://github.com/Seeed-Studio/ModelAssistant/releases/tag/model_zoo) |

:::tip
更多模型请前往 [SSCMA Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo)
:::

## 贡献

- 如果你在这些例子中发现了问题，或者希望提交一个增强请求，请使用 [GitHub Issue](https://github.com/Seeed-Studio/ModelAssistant)。

- 对于 ESP-IDF 相关的问题请参考 [ESP-IDF](https://github.com/espressif/esp-idf)。

- 对于 TensorFlow Lite Micro 相关的信息请参考 [TFLite-Micro](https://github.com/tensorflow/tflite-micro)。

- 对于 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 相关的信息请参考 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant)。

## 许可

这些例子使用 ESP-IDF，它是在 [Apache 2.0 许可](https://github.com/espressif/esp-idf/blob/master/LICENSE)下发布的。

TensorFlow 库代码和第三方代码包含他们自己的许可证，在 [TFLite-Micro](https://github.com/tensorflow/tflite-micro) 中有说明。
