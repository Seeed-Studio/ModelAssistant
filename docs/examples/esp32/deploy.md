# Deploying EdgeLab on Espressif Chips

This example is a tutorial for deploying the models from [EdgeLab](https://github.com/Seeed-Studio/Edgelab/) to Espreessif chipsets, based on [ESP-IDF](https://github.com/espressif/esp-idf) and [Tensorflow Lite Micro](https://github.com/tensorflow/tflite-micro) implementations.


## Prerequisites

### Hardware

- A Linux or macOS computer

- An ESP32-S3 development board with a camera (e.g. [Seeed Studio XIAO](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html))

- A USB cable

### Install the ESP-IDF

EdgeLab requires ESP-IDF `4.4.x` for deployment in ESP32, please refer to the following tutorial [ESP-IDF Get Started Guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) to install and configure the toolchain of ESP-IDF.

After completed the ESP-IDF installation, please double-check [IDF Environment Config Variables](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables) is finished:

* The `IDF_PATH` environment variable is set.

* Make sure that tools like `idf.py` and Xtensa-ESP32 (e.g. `xtensa-esp32-elf-gcc`) are included in `$PATH`.

::: tip

We do not recommend that you configure ESP-IDF in a virtual environment, you can use the following command to exit a virtual environment (use the command multiple times to exit nested virtual environments).

```sh
conda deactivate
```

In addition, if your ESP-IDF is not configured in a virtual environment, any operations related to ESP-IDF, such as calls to `idf.py`, should be done in a non-virtual environment.

:::

### Get Examples and Submodules

**Go to the root directory of the EdgeLab project** and run the following command to get the examples and its submodules.

```sh
# clone Seeed-Studio/edgelab-example-esp32 to example/esp32
git clone https://github.com/Seeed-Studio/edgelab-example-esp32 example/esp32

# go to example/esp32, pull the submodules, and return to the EdgeLab project root directory
pushd example/esp32
git submodule init
git submodule update
popd
```

::: warning

You need to complete the installation and configuration of EdgeLab first. If you have not installed EdgeLab yet, you can refer to [EdgeLab Installation Guide](../../introduction/installation.md).

:::


## Prepare the Model

Before you start compiling and deploying, you need to prepare the models that need to be deployed according to the actual application scenarios. Therefore, you may need to go through the steps of selecting a model or neural network, customizing the dataset, exporting or transforming the model, etc.

To help you understand the process in a more organized way, we have written complete documentation for different application scenarios.

- [**ESP32 Mask Detection**](./mask_detection.md)

- [**ESP32 Meter Reader**](./meter_reader.md)


::: warning

Before [Compile and Deploy](#compile-and-deploy), you need to prepare the appropriate model.

:::


## Compile and Deploy

### Compile

1. Go to the root directory of the EdgeLab project and run the following command to access the examples directory ``examples/esp32``.

```sh
cd examples/esp32 # EdgeLab/examples/esp32
```

2. Set `IDF_TARGET` to `esp32s3`.

```sh
idf.py set-target esp32s3
```

3. Compile the example.

```sh
idf.py build
```

### Deploy

1. Connect the ESP32 MCU to the computer and determine the serial port path. Under Linux, you can use the following command to check the currently available serial ports (the path to the serial port is typically `/dev/ttyUSB0` for newly connected ESP32 devices on Linux):

```sh
lsusb -t
ls /dev | grep tty
```

2. Flash the firmware (replace `<TARGET_SERIAL_PORT>` with the ESP32 serial port path):

```sh
idf.py --port <TARGET_SERIAL_PORT> flash
```

3. Monitor serial output and wait for MCU reboot (replace `<TARGET_SERIAL_PORT>` with the ESP32 serial path):

```sh
idf.py --port <TARGET_SERIAL_PORT> monitor
```

::: tip

The two commands to flash the firmware and monitor the serial port can be combined.

```sh
idf.py --port <TARGET_SERIAL_PORT> flash monitor
```

Use `Ctrl+]` to exit the monitor serial output screen.

:::


### Performance Profile

The performance of EdgeLab related models, measured on different chips, is summarized in the following table.

| Target | Model | Dataset | Input Resolution | Peak RAM | Inferencing  Time | F1 Score | Link |
|--|--|--|--|--|--|--|--|
| ESP32-S3 | Meter | [Custom Meter](https://files.seeedstudio.com/wiki/Edgelab/meter.zip) | 112x112 (RGB) | 320KB | 380ms | 97% | [pfld_meter_int8.tflite](https://github.com/Seeed-Studio/EdgeLab/releases) |
| ESP32-S3 | Fomo | [COCO MASK](https://files.seeedstudio.com/wiki/Edgelab/coco_mask.zip) | 96x96 (GRAY) | 244KB | 150ms | 99.5% | [fomo_mask_int8.tflite](https://github.com/Seeed-Studio/EdgeLab/releases) |



## Contribute

- If you find any issues in these examples, or wish to submit an enhancement request, please use [GitHub Issue](https://github.com/Seeed-Studio/EdgeLab).

- For ESP-IDF related issues please refer to [ESP-IDF](https://github.com/espressif/esp-idf).

- For information about TensorFlow Lite Micro, please refer to [TFLite-Micro](https://github.com/tensorflow/tflite-micro).

- For EdgeLab related information, please refer to [EdgeLab](https://github.com/Seeed-Studio/Edgelab/).


## Licensing

These examples are released under the [MIT License](../../community/licenses.md).

These examples use ESP-IDF, which is released under the [Apache 2.0 License](https://github.com/espressif/esp-idf/blob/master/LICENSE).

The TensorFlow library code and third-party code contain their own licenses, which are described in [TFLite-Micro](https://github.com/tensorflow/tflite-micro).
