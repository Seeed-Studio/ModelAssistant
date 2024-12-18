# Deploying Models on Espressif Chips

This example is a tutorial for deploying models included in [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) on Espressif chips, with the deployment work based on [ESP-IDF](https://github.com/espressif/esp-idf) and [Tensorflow Lite Micro](https://github.com/tensorflow/tflite-micro).

## Prerequisites

### Hardware

- A Linux or macOS computer

- An ESP32-S3 development board with a camera (e.g., [Seeed Studio XIAO](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html))

- A USB data cable

### Installing ESP-IDF

The deployment of models included in [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) on ESP32 requires ESP-IDF `5.1.x`. Please refer to the following tutorial [ESP-IDF Get Started Guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) to install and configure the toolchain and ESP-IDF.

After successfully installing ESP-IDF, please confirm again whether the [IDF environment variables are set up](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables):

- The `IDF_PATH` environment variable is set.

- Ensure that tools such as `idf.py` and Xtensa-ESP32 (e.g., `xtensa-esp32-elf-gcc`) are included in `$PATH`.

:::tip

We do not recommend configuring ESP-IDF in a virtual environment. You can use the following command to exit the virtual environment (can be used multiple times to exit nested virtual environments):

```sh
conda deactivate
```

Additionally, if your ESP-IDF is not configured in a virtual environment, any operations related to ESP-IDF, such as calls to `idf.py`, should be performed outside of the virtual environment.

:::

### Obtaining Examples and Submodules

**Navigate to the root directory of the [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) project** and run the following commands to obtain the examples and submodules.

```sh
git clone https://github.com/Seeed-Studio/sscma-example-esp32 -b 1.0.0 examples/esp32 && \
pushd examples/esp32 && \
git submodule init && \
git submodule update && \
popd
```

:::warning

You need to complete the installation and configuration of [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) in advance. If you have not installed [SSCMA](https://github.com/Seeed-Studio/ModelAssistant), please refer to the [SSCMA Installation Guide](../../introduction/installation).

:::

## Preparing the Model

Before starting to compile and deploy, you need to prepare the model you want to deploy according to the actual application scenario. Therefore, you may need to go through steps such as selecting a model or neural network, customizing a dataset, exporting or converting a model, etc.

To help you understand the process more systematically, we have written complete documents for different application scenarios [SSCMA - Model Training and Export](../training/overview.md).

:::warning

Before [compiling and deploying](#compiling-and-deploying), you need to prepare the corresponding model in advance.

:::

## Compiling and Deploying

### Compiling Routines

1. Navigate to the root directory of the [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) project and run the following command to enter the example directory `examples`:

```sh
cd examples/<examples>
```

2. Set `IDF_TARGET` to `esp32s3`:

```sh
idf.py set-target esp32s3
```

3. Compile the routine:

```sh
idf.py build
```

### Deploying Routines

1. Connect the ESP32 MCU to the computer and determine the serial port path of the ESP32. On Linux, you can use the following command to check the currently available serial ports (for newly connected ESP32 devices on Linux, the serial port path is generally `/dev/ttyUSB0`):

```sh
lsusb -t && \
ls /dev | grep tty
```

2. Flash the firmware (replace `<TARGET_SERIAL_PORT>` with the serial port path of the ESP32):

```sh
idf.py --port <TARGET_SERIAL_PORT> flash
```

3. Monitor the serial output and wait for the MCU to restart (replace `<TARGET_SERIAL_PORT>` with the serial port path of the ESP32):

```sh
idf.py --port <TARGET_SERIAL_PORT> monitor
```

:::tip

The two commands for flashing the firmware and monitoring the serial output can be combined:

```sh
idf.py --port <TARGET_SERIAL_PORT> flash monitor
```

Use `Ctrl+]` to exit the serial output monitoring interface.

:::

### Performance Overview

By measuring on different chips, the performance of models related to [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) is summarized in the table below.

| Target | Model | Dataset | Input Resolution | Peak RAM | Inferencing Time | F1 Score | Link |
|--|--|--|--|--|--|--|--|
| ESP32-S3 | Meter | [Custom Meter](https://files.seeedstudio.com/sscma/datasets/meter.zip) | 112x112 (RGB) | 320KB | 380ms | 97% | [pfld_meter_int8.tflite](https://github.com/Seeed-Studio/ModelAssistant/releases/tag/model_zoo) |
| ESP32-S3 | Fomo | [COCO MASK](https://files.seeedstudio.com/sscma/datasets/coco_mask.zip) | 96x96 (GRAY) | 244KB | 150ms | 99.5% | [fomo_mask_int8.tflite](https://github.com/Seeed-Studio/ModelAssistant/releases/tag/model_zoo) |

:::tip
For more models, please visit [SSCMA Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo)

:::

## Contributing

- If you find any issues in these examples or wish to submit an enhancement request, please use [GitHub Issue](https://github.com/Seeed-Studio/ModelAssistant).

- For ESP-IDF related issues, please refer to [ESP-IDF](https://github.com/espressif/esp-idf).

- For TensorFlow Lite Micro related information, please refer to [TFLite-Micro](https://github.com/tensorflow/tflite-micro).

- For [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) related information, please refer to [SSCMA](https://github.com/Seeed-Studio/ModelAssistant).

## License

These examples use ESP-IDF, which is released under the [Apache 2.0 License](https://github.com/espressif/esp-idf/blob/master/LICENSE).

TensorFlow library code and third-party code include their own licenses, which are explained in [TFLite-Micro](https://github.com/tensorflow/tflite-micro).
