# VAE 模型训练

本节将介绍如何 VAE 模型进行异常检测任务的训练。VAE 模型是一种生成式模型，它可以学习数据的分布，并生成与原始数据相似的数据。在异常检测任务中，我们可以利用 VAE 模型学习正常数据的分布，然后通过计算异常分数来判断数据是否异常。

## 数据准备

使用 VAE 模型进行异常检测前需要手动采集数据，这里以三轴陀螺仪为例，使用 ESP32 进行数据采集。

### 编译固件

首先需要编译 ESP32 的固件，可以参考 [SSCMA - 在 Espressif 芯片上部署模型](../deploy/xiao_esp32s3)。

我们提供了用于数据采集的源码，您可以访问 [SSCMA Example ESP32 - GADNN](https://github.com/Seeed-Studio/sscma-example-esp32/blob/dev/examples/gyro_anomaly_detection_nn/main/app_main.cpp) 获取源码。

接下来检查示例程序，可以看到示例默认以 QMA7981 三轴陀螺仪为传感器，采样频率为 100Hz，修改宏 `GYRO_SAMPLE_MODE` 为 1 即可开启数据采集模式，采集的数据将由 ESP32 通过串口发送到 PC 端。

### 数据采集

将 ESP32 与 PC 通过 USB 线连接，假设设备被挂载于 `/dev/ttyACM0`（Linux）或 `COM3`（Windows），运行以下命令进行数据采集：

```sh
python3 tools/dataset_tool/read_serial.py \
    -sr 115200 \
    -p /dev/ttyACM0 \
    -f datasets/accel_3axis.csv
```

:::tip

采集数据时，确保设备稳定放置，避免数据采集过程中设备移动，如果需要检测设备移动时的异常，可以在数据采集时进行移动，不过需要确保采集期间设备的运动状态为正常运动状态。

:::

### 数据预处理

数据采集完成后，我们需要对数据进行预处理:

```sh
python tools/dataset_tool/signal_data_processing.py \
    -t train \
    -f datasets/accel_3axis.csv
```

预处理完成后，数据集将被保存在 `datasets/accel_3axis` 目录下。


## 模型训练

在此我们以 `vae_mirophone.py` 为例，展示如何使用 SSCMA 进行 FOMO 模型训练。

```sh
python3 tools/train.py \
    configs/anomaly/vae_mirophone.py \
    --cfg-options \
    data_root=$(pwd)/datasets/accel_3axis/
```

- `configs/anomaly/vae_mirophone.py`: 指定配置文件，定义模型和训练设置。
- `--cfg-options`: 用于指定额外的配置选项。
    - `data_root`: 设定数据集的根目录。


## 模型导出及验证

在训练过程中，您可以随时查看训练日志、导出模型并验证模型的性能，部分模型验证中输出的指标在训练过程中也会显示，因此在这一部分我们会先介绍如何导出模型，然后阐述如何验证导出后模型的精度。

### 导出模型

这里我们以导出 TFLite 模型为例，您可以使用以下命令导出不同精度的 TFLite 模型：

```sh
python3 tools/export.py \
    configs/anomaly/vae_mirophone.py \
    work_dirs/epoch_100.pth \    
    --cfg-options \
    data_root=$(pwd)/datasets/accel_3axis/ \
    --imgsz 32 32 \
    --format tflite \
    --image_path $(pwd)/datasets/accel_3axis/train
```

:::warning

我们建议在训练和导出时使用相同的分辨率，在当前情况下，使用不同的分辨率训练和导出时，可能导致模型精度降低或完全丢失。

:::

:::tip

在导出过程中，可能需要网络环境以安装某些依赖，如果无法访问到互联网，请确保当前的 Python 环境中已经安装了以下依赖：

```
tensorflow
hailo_sdk_client
onnx
onnx2tf
tf-keras
onnx-graphsurgeon
sng4onnx
onnxsim
```

此外，`onnx2tf` 在运行时可能还需要下载 calibration 相关的数据，您可以参考以下链接将其提前下载到 SSCMA 的根目录。

```sh
wget https://github.com/PINTO0309/onnx2tf/releases/download/1.20.4/calibration_image_sample_data_20x128x128x3_float32.npy \
    -O calibration_image_sample_data_20x128x128x3_float32.npy
```

:::


### 验证模型

在导出模型后，您可以使用以下命令验证模型的性能：

```sh
python3 tools/test.py \
    configs/anomaly/vae_mirophone.py \
    work_dirs/epoch_100_int8.tflite \    
    --cfg-options \
    data_root=$(pwd)/datasets/accel_3axis/ 
```

### QAT

QAT（量化感知训练）是一种在模型训练过程中模拟量化操作，让模型逐步适应量化误差，从而在量化后仍能保持较高精度的方法。SSCMA 支持 QAT，您可以参考以下方法得到 QAT 的模型，并再次验证。

```sh
python3 tools/quantization.py \
    configs/anomaly/vae_mirophone.py \
    work_dirs/epoch_100.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/accel_3axis/
```

QAT 训练完毕后，会自动导出量化后的模型，其存放路径为 `out/qat_model_test.tflite`，您可以使用以下命令对其进行验证：

```sh
python3 tools/test.py \
    configs/anomaly/vae_mirophone.py \
    out/qat_model_test.tflite \    
    --cfg-options \
    data_root=$(pwd)/datasets/accel_3axis/ 
```


## 模型部署

首先，使用 `xxd` 工具将 TFLite 模型转换为 C 数组：

```sh
cp out/qat_model_test.tflite out/nn_model_tflite && \
xxd -i out/nn_model_tflite > out/nn_model.h
```

然后，将生成的 `nn_model.h` 头文件拷贝到 ESP32 项目的 `sscma-example-esp32/examples/gyro_anomaly_detection_nn/main` 目录下，替换原有的 `nn_model.h` 文件。

最后，修改宏 `GYRO_SAMPLE_MODE` 为 0，重新编译 ESP32 项目，即可将模型部署到 ESP32 上。