# Deploying EdgeLab on Grove - Vision AI

This example is a tutorial for deploying the models from [EdgeLab](https://github.com/Seeed-Studio/Edgelab/) to Grove - Vision AI module, based on the [Synopsys GUN Toolchain](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain) and [Tensorflow Lite Micro](https://github.com/tensorflow/tflite-micro) implementations.


## Prerequisites

### Hardware

- A computer with Linux or Windows (WSL) (this example uses [Ubuntu 20.04](https://releases.ubuntu.com/focal/))

- A [Grove - Vision AI module](https://www.seeedstudio.com/Grove-Vision-AI-Module-p-5457.html)

- A USB cable

### Installing Synopsys GUN Toolchain

Grove - Vision AI uses the [Himax HX6537](https://www.himax.com.tw/zh/products/intelligent-sensing/always-on-smart-sensing/) chip, where we need to install the [ Synopsys GUN Toolchain](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain) in order to cross-compile the firmware afterwards, the installation is divided into the following steps.

1. First, download the pre-compiled toolchain from [Synopsys GUN Toolchain - GitHub Releases](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/releases/).

    ```sh
    # download the arc-2020.09-release version to the home directory ~/
    wget https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/releases/download/arc-2020.09-release/arc_gnu_2020.09_prebuilt_elf32_le_linux_install.tar.gz -P ~/

    # extract the downloaded toolchain to the home directory ~/
    tar -zxvf ~/arc_gnu_2020.09_prebuilt_elf32_le_linux_install.tar.gz --directory ~/
    ```

2. Then, specify the directory of the Synopsys GUN Toolchain executable in the PATH and add it to `~/.bashrc` to facilitate automatic import when shell starts.

    ```sh
    echo 'export PATH="$HOME/arc_gnu_2020.09_prebuilt_elf32_le_linux_install/bin:$PATH" # Synopsys GUN Toolchain' >> ~/.bashrc
    ```

    ::: tip

    If you are using Zsh or other Shells, the above `~/.bashrc` should be adjusted accordingly.

    :::

### Get Examples and the SDK

**Please go to the root of the EdgeLab project**, then run the following command to get the examples and download the SDK.

```sh
# clone Seeed-Studio/edgelab-example-vision-ai to example/grove
git clone https://github.com/Seeed-Studio/edgelab-example-vision-ai example/grove

# go to example/grove and download the default TFLite models and SDK
pushd example/grove
make download
popd
```

::: tip

If you have not installed [Make](https://www.gnu.org/software/make/), on Linux distributions that use APT as the default package manager, you can install it with the following command.

```sh
# update source
sudo apt-get update

# install make
sudo apt-get install make -y
```

In addition, we recommend that you complete the installation and configuration of EdgeLab in advance. If you have not installed EdgeLab yet, you can refer to [EdgeLab Installation Guide](../../introduction/installation.md).

:::


## Prepare the Model

Before you start compiling and deploying, you need to prepare the models to be deployed according to the actual application scenarios. Models are included in the default Grove - Vision AI SDK, or you can try to train different models yourself.

Therefore, you may need to go through steps such as model or neural network selection, customizing the dataset, training, exporting and converting the model.

To help you understand the process in a more organized way, we have written complete documentation for different application scenarios.

- [**Grove Mask Detection**](./mask_detection.md)

- [**Grove Meter Reader**](./meter_reader.md)


::: warning

Before [Compile and Deploy](#compile-and-deploy), you need to prepare the appropriate model.

:::


## Compile and Deploy

### Compile the Firmware and Model Firmware

1. First, please go to the root directory of the EdgeLab project and run the following command to access the example directory `examples/grove`.

    ```sh
    cd examples/grove # EdgeLab/examples/grove
    ```

2. Second, choose the compilation parameters according to **selected model** and compile them, the optional parameters are `fomo`, `meter`, etc.

    ::: code-group

    ```sh [fomo]
    # grove mask detection
    make HW=grove_vision_ai APP=fomo
    ```

    ```sh [meter]
    # grove meter reader
    make HW=grove_vision_ai APP=meter
    ```

    :::

    ::: tip

    You can view all optional parameters for APP using the following command.

    ```sh
    ls examples # EdgeLab/examples/grove/examples
    ```

    After compilation, a binary file named `output.img` will be generated in the `tools/image_gen_cstm/output` directory.

    :::

3. Third, generate the UF2 firmware image.

    ```sh
    python3 tools/ufconv/uf2conv.py -t 0 -c tools/image_gen_cstm/output/output.img -o firmware.uf2
    ```

4. Last, generate UF2 model image from TFLite model.

    ```sh
    python3 tools/ufconv/uf2conv.py -t 1 -c <TFLITE_MODEL_PATH> -o model.uf2
    ```

    ::: tip

    You need to replace `<TFLITE_MODEL_PATH>` with the path to the TFLite model obtained in the [Prepare the Model](#prepare-the-model) step. You can also use the pre-trained model, which is located in the `model_zone` directory, and simply copy its path.

    Note that the **model type** selected should be consistent with the selection which you have made in [Compile Firmware and Model Firmware - Step 2](#compile-the-firmware-and-model-firmware).

    :::

### Deployment Routines

The deployment process of Grove - Vision AI is divided into two main steps, which need to be executed in strict order.

1. **Flash `firmware.uf2` firmware image** and reboot or reconnect.

2. **Flash `model.uf2` model firmware image** and reboot or reconnect.

3. Connect Grove - Vision AI to the computer again via the USB cable and use a browser which supports [WebUSB API](https://developer.mozilla.org/en-US/docs/Web/API/WebUSB_API) such as [Google Chrome]( https://www.google.com/chrome/), then access the [Grove Vision AI Console](https://files.seeedstudio.com/grove_ai_vision/index.html).

4. In the browser interface and [Grove Vision AI Console](https://files.seeedstudio.com/grove_ai_vision/index.html), select **Grove AI** in the pop-up window and click **Connect** button to connect.

::: tip

**How to flash the firmware and model?**

1. Connect Grove - Vision AI module to your computer via USB Type-C cable.

2. **Double click** on the **BOOT button** of Grove - Vision AI module to put it into DFU mode.

3. Wait for a moment and a storage device named **GROVEAI** or **VISIONAI** will appear.

4. Copy the UF2 firmware to the root directory of the storage device. When the device disappears, the flashing process is complete.

:::


### Performance Profile

The performance of EdgeLab related models, measured on different chips, is summarized in the following table.

| Target | Model | Dataset | Input Resolution | Peak RAM | Inferencing Time | F1 Score | Link |
|--|--|--|--|--|--|--|--|
| Grove Vision AI | Meter | [Custom Meter](https://files.seeedstudio.com/wiki/Edgelab/meter.zip) | 112x112 (RGB) | 320KB | 500ms | 97% | [pfld_meter_int8.tflite](https://github.com/Seeed-Studio/EdgeLab/releases) |
| Grove Vision AI | Fomo | [COCO MASK](https://files.seeedstudio.com/wiki/Edgelab/coco_mask.zip) | 96x96 (GRAY) | 244KB | 150ms | 99.5% | [fomo_mask_int8.tflite](https://github.com/Seeed-Studio/EdgeLab/releases) |


## Contribute

- If you find any issues in these examples, or wish to submit an enhancement request, please use [GitHub Issue](https://github.com/Seeed-Studio/EdgeLab).

- For Synopsys GUN Toolchain related issues please refer to [Synopsys GUN Toolchain](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain).

- For information about TensorFlow Lite Micro, please refer to [TFLite-Micro](https://github.com/tensorflow/tflite-micro).

- For EdgeLab related information, please refer to [EdgeLab](https://github.com/Seeed-Studio/Edgelab/).


## Licensing

These examples are released under the [MIT License](../../community/licenses.md).

For Synopsys GUN Toolchain, it is released under the [GPLv3 License](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/blob/arc-releases/COPYING).

The TensorFlow library code and third-party code contain their own licenses, which are described in [TFLite-Micro](https://github.com/tensorflow/tflite-micro).
