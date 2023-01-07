# Edgelab for Espressif Chipsets

- [Introduction](#introduction)
- [How to Install](#how-to-install)
  * [Install the ESP IDF](#install-the-esp-idf)
- [Build the example](#build-the-example)

## Introduction

Edgelab examples for Espressif chipsets is a collection of examples for Espressif chipsets. It is based on the [ESP-IDF](https://github.com/espressif/esp-idf) and [tensorflow lite micro](https://github.com/tensorflow/tflite-micro).


## How to Install

### Install the ESP IDF

Follow the instructions of the
[ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)
to setup the toolchain and the ESP-IDF itself.

The next steps assume that this installation is successful and the
[IDF environment variables are set](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables). Specifically,
* the `IDF_PATH` environment variable is set
* the `idf.py` and Xtensa-esp32 tools (e.g., `xtensa-esp32-elf-gcc`) are in `$PATH`

### get submodules
cd to the root directory of the project and run the following command to get the submodules:

```
git submodule init
git submodule update examples/esp32/compoents/esp-nn
git submodule update examples/esp32/compoents/esp32-camera
```


## Build the example

Go to example directory (`examples/<example_name>`) and build the example.

Set the IDF_TARGET (For ESP32-S3 target, IDF version `release/v4.4` is needed)

```
idf.py set-target esp32s3
```

To build this, run:

```
idf.py build
```

### Load and run the example

To flash (replace `/dev/ttyUSB0` with the device serial port):
```
idf.py --port /dev/ttyUSB0 flash
```

Monitor the serial output:
```
idf.py --port /dev/ttyUSB0 monitor
```

Use `Ctrl+]` to exit.

The previous two commands can be combined:
```
idf.py --port /dev/ttyUSB0 flash monitor
```

  - Please follow example READMEs for more details.


### Performance Comparison

A quick summary of ESP-NN optimisations, measured on various chipsets:

|   Target  |   TFLite Micro Example  |  with ESP-NN | CPU Freq  |
| --------- | ----------------------- | -------------| ----------|
| ESP32-S3  |   Meter                 |     550ms    |   240MHz  |


## Contributing
- If you find an issue in these examples, or wish to submit an enhancement request, please use the Issues section on Github.
- For ESP-IDF related issues please use [esp-idf](https://github.com/espressif/esp-idf) repo.
- For TensorFlow related information use [tflite-micro](https://github.com/tensorflow/tflite-micro) repo.

## License

These examples are covered under MIT License.

These examples use the ESP-IDF which is covered under Apache License 2.0.

TensorFlow library code and third_party code contains their own license specified under respective [repos](https://github.com/tensorflow/tflite-micro).