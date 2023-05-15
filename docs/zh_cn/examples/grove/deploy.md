# 在 Grove - Vision AI 上部署 EdgeLab

本示例为 [EdgeLab](https://github.com/Seeed-Studio/Edgelab/) 包含的模型在 Grove - Vision AI 模块的部署教程，部署工作基于 [Synopsys GUN Toolchain](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain) 和 [Tensorflow Lite Micro](https://github.com/tensorflow/tflite-micro) 实现。


## 先决条件

### 硬件

- 一台使用 Linux 或者 Windows (WSL) 的计算机 (本示例使用 [Ubuntu 20.04](https://releases.ubuntu.com/focal/))

- 一个 [Grove - Vision AI 模块](https://www.seeedstudio.com/Grove-Vision-AI-Module-p-5457.html)

- 一根 USB 数据线

### 安装 Synopsys GUN Toolchain

Grove - Vision AI 使用了 [Himax HX6537](https://www.himax.com.tw/zh/products/intelligent-sensing/always-on-smart-sensing/) 芯片，在这里我们需要安装 [Synopsys GUN Toolchain](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain) 以便之后交叉编译生成固件，其安装分为以下步骤:

1. 首先，从 [Synopsys GUN Toolchain - GitHub Releases](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/releases/) 下载预编译好的工具链。

    ```sh
    # 在这里我们下载 arc-2020.09-release 版本到用户主目录 ～/
    wget https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/releases/download/arc-2020.09-release/arc_gnu_2020.09_prebuilt_elf32_le_linux_install.tar.gz -P ~/

    # 解压下载好的工具链到用户主目录 ～/
    tar -zxvf ~/arc_gnu_2020.09_prebuilt_elf32_le_linux_install.tar.gz --directory ~/
    ```

2. 然后，在 PATH 中指明 Synopsys GUN Toolchain 的可执行文件目录，并将其添加到 `~/.bashrc`，方便在 Shell 启动时自动导入。

    ```sh
    echo 'export PATH="$HOME/arc_gnu_2020.09_prebuilt_elf32_le_linux_install/bin:$PATH" # Synopsys GUN Toolchain' >> ~/.bashrc
    ```

    ::: tip

    如果您使用 Zsh 或其它 Shell，上述的 `~/.bashrc` 也应作出相应的调整。 

    :::

### 获取示例并配置 SDK

**进入 EdgeLab 项目的根目录**，运行下面的命令来获取示例和。

```sh
# 克隆 Seeed-Studio/edgelab-example-vision-ai 到 example/grove
git clone https://github.com/Seeed-Studio/edgelab-example-vision-ai example/grove

# 进入 example/grove，下载默认的 TFLite 模型和库数据
pushd example/grove
make download
popd
```

::: tip

如果您还没有安装 [Make](https://www.gnu.org/software/make/)，在使用 APT 为默认包管理器的 Linux 发行版操作系统上，您可以参考如下命令安装:

```sh
# 更新源
sudo apt-get update

# 安装 make
sudo apt-get install make -y
```

此外，我们建议您提前完成 EdgeLab 的安装与配置。如果您还没有安装 EdgeLab，可以参考 [EdgeLab 安装指南](../../introduction/installation.md)。

:::


## 准备模型

在开始编译和部署之前，您需要先根据实际应用场景，准备好需要部署的模型。默认的 Grove - Vision AI SDK 中包含了模型，您也可以尝试自行训练不同的模型。

因此，您可能需要经历模型或神经网络的选择、自定义数据集、训练、导出或转换模型等步骤。

为了让您更有条理地理解该过程，我们针对不同的应用场景编写了完整的文档:

- [**Grove 口罩检测**](./mask_detection.md)

- [**Grove 表计读数**](./meter_reader.md)


::: warning

在[编译和部署](#编译和部署)前，您需要提前准备好相应的模型。

:::


## 编译和部署

### 编译固件和模型固件

1. 进入 EdgeLab 项目的根目录，运行以下命令进入示例目录 `examples/grove`:

    ```sh
    cd examples/grove # EdgeLab/examples/grove
    ```

2. 根据**模型种类**选择编译参数并编译，可选的参数有 `fomo`、`meter` 等:

    ::: code-group
    
    ```sh [fomo]
    # Grove 口罩检测
    make HW=grove_vision_ai APP=fomo
    ```
    
    ```sh [meter]
    # Grove 表计读数
    make HW=grove_vision_ai APP=meter
    ```
    
    :::
    
    ::: tip
    
    关于 APP 的所有可选参数在可以使用以下命令查看:
    
    ```sh
    ls examples # EdgeLab/examples/grove/examples
    ```
    
    编译完成后，会在 `tools/image_gen_cstm/output` 目录下产生名为 `output.img` 的二进制文件。
    
    :::

3. 生成 UF2 固件镜像:

    ```sh
    python3 tools/ufconv/uf2conv.py -t 0 -c tools/image_gen_cstm/output/output.img -o firmware.uf2
    ```

4. 从 TFLite 模型生成 UF2 模型镜像:

    ```sh
    python3 tools/ufconv/uf2conv.py -t 1 -c <TFLITE_MODEL_PATH> -o model.uf2
    ```
    
    ::: tip

    您需要将 `<TFLITE_MODEL_PATH>` 替换为在[准备模型](#准备模型)步骤中取得的 TFLite 模型的路径。您也可以使用预训练好的模型，其位于 `model_zone` 目录下，只需要复制其路径即可。
    
    需要注意的是，选取的**模型类型**应与[编译固件和模型固件 - 第 2 步](#编译固件和模型固件)中的选择保持一致。

    :::

### 部署例程

Grove - Vision AI 的部署流程主要分为两个步骤，这两个步骤需要严格按顺序执行:

1. **刷写 `firmware.uf2` 固件镜像**并重启或重新连接。

2. **刷写 `model.uf2` 模型固件镜像**并重启或重新连接。

3. 将 Grove - Vision AI 通过数据线再次连接至计算机，使用支持 [WebUSB API](https://developer.mozilla.org/en-US/docs/Web/API/WebUSB_API) 的浏览器如 [Google Chrome](https://www.google.com/chrome/) 等，访问 [Grove Vision AI 控制台](https://files.seeedstudio.com/grove_ai_vision/index.html)。

4. 在浏览器界面和 [Grove Vision AI 控制台](https://files.seeedstudio.com/grove_ai_vision/index.html)，在弹出窗口中选择 **Grove AI** 然后点击 **Connect** 进行连接。


::: warning

在烧录固件之前，您必须检查设备 Bootloader 的版本是否与固件版本匹配。如果不匹配，您需要先刷写 Bootloader 固件，然后再刷写固件。

:::

::: tip

**如何烧录固件与模型？**

1. 通过 USB Type-C 线缆将 Grove - Vision AI 连接到计算机。

2. **双击** Grove - Vision AI 的 **BOOT 按键**，使得其进入 DFU 模式。

3. 稍等片刻，会出现一个名为 **GROVEAI** 或 **VISIONAI** 的存储设备。

4. 将要刷入的 UF2 固件复制到该储存设备的根目录下，当设备消失时，表明烧录完成。

:::


### 性能简介

通过在不同的芯片上测量，对 EdgeLab 相关模型的性能总结如下表所示。

| Target | Model | Dataset | Input Resolution | Peak RAM | Inferencing Time | F1 Score | Link |
|--|--|--|--|--|--|--|--|
| Grove Vision AI | Meter | [Custom Meter](https://files.seeedstudio.com/wiki/Edgelab/meter.zip) | 112x112 (RGB) | 320KB | 500ms | 97% | [pfld_meter_int8.tflite](https://github.com/Seeed-Studio/EdgeLab/releases) |
| Grove Vision AI | Fomo | [COCO MASK](https://files.seeedstudio.com/wiki/Edgelab/coco_mask.zip) | 96x96 (GRAY) | 244KB | 150ms | 99.5% | [fomo_mask_int8.tflite](https://github.com/Seeed-Studio/EdgeLab/releases) |


### Check BootLoader Version

你可能需要检测 BootLoader 的版本是否需要更新，以决定是否应该进行更新。检查版本号的方法如下。

- 双击 `BOOT` 按钮，等待可移动驱动器挂载
- 打开可移动驱动器中的 `INFO_UF2.TXT`

<div style={{textAlign:'center'}}><img src="https://raw.githubusercontent.com/Seeed-Studio/Seeed_Arduino_GroveAI/master/assert/q2.png" style={{width:800, height:'auto'}}/></div>

你可以看到，图片中的第三行是 **BootLoader** 的版本号。如果它与我们发布的版本号相同，你就不需要更新 **BootLoader**。


### Update BootLoader

如果您的 `Grove Vision AI` 不被计算机识别，表现为没有端口号，那么您可能需要更新 **BootLoader**。

- **第1步**。在Windows电脑上下载 `BootLoader.bin` 文件。

请在下面的链接中下载最新版本的 **BootLoader** 文件。**BootLoader** 的名称通常是 `tinyuf2-grove_vision_ai_vx.x.bin`。

<div class="github_container" style={{textAlign: 'center'}}>
    <a class="github_item" href="https://github.com/Seeed-Studio/Seeed_Arduino_GroveAI/releases">
    <strong><span><font color={'FFFFFF'} size={"4"}> Download the Firware</font></span></strong> <svg aria-hidden="true" focusable="false" role="img" className="mr-2" viewBox="-3 10 9 1" width={16} height={16} fill="currentColor" style={{textAlign: 'center', display: 'inline-block', userSelect: 'none', verticalAlign: 'text-bottom', overflow: 'visible'}}><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z" /></svg>
    </a>
</div>

这是控制 **BL702** 芯片的固件，它建立了计算机和 **Himax** 芯片之间的连接。最新版本的 **BootLoader** 现在已经解决了 `Vision AI` 无法被 Mac 和 Linux 识别的问题。

- **Step 2**. 下载并打开 [**BLDevCube.exe**](https://files.seeedstudio.com/wiki/Grove_AI_Module/BouffaloLabDevCube-1.6.6-win32.rar) 软件，选择 **BL702/704/706** ，然后点击**完成**。

<div style={{textAlign:'center'}}><img src="https://files.seeedstudio.com/wiki/Grove_AI_Module/GroveAI01a.png" style={{width:300, height:'auto'}}/></div>

- **Step 3**. 点击**查看**，先选择**MCU**。移到**图像文件**，点击**浏览**，选择你刚下载的固件。

<div style={{textAlign:'center'}}><img src="https://files.seeedstudio.com/wiki/Grove_AI_Module/1.png" style={{width:800, height:'auto'}}/></div>

- **Step 4**. 确保没有其他设备连接到PC上。然后按住模块上的启动按钮，将其连接到PC上。

  <div style={{textAlign:'center'}}><img src="https://files.seeedstudio.com/wiki/Grove_AI_Module/GroveAI05.png" style={{width:600, height:'auto'}}/></div>

  我们可以看到模块背面的5V灯和3.3V的LED灯正在点亮，然后松开启动按钮。

  <div style={{textAlign:'center'}}><img src="https://files.seeedstudio.com/wiki/Grove_AI_Module/GroveAI06.png" style={{width:600, height:'auto'}}/></div>

- **Step 5**. 回到PC上的BLDevCube软件，点击 **Refresh**，选择一个合适的端口。然后点击  **Open UART**，并将 **Chip Erase** 设置为 **True** ，然后点击 **Creat&Program** ，等待过程完成。


## 贡献

- 如果你在这些例子中发现了问题，或者希望提交一个增强请求，请使用 [GitHub Issue](https://github.com/Seeed-Studio/EdgeLab)。

- 对于 Synopsys GUN Toolchain 相关的问题请参考 [Synopsys GUN Toolchain](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain)。

- 对于 TensorFlow Lite Micro 相关的信息请参考 [TFLite-Micro](https://github.com/tensorflow/tflite-micro)。

- 对于 EdgeLab 相关的信息请参考 [EdgeLab](https://github.com/Seeed-Studio/Edgelab/)。


## 许可

这些例子是在 [MIT 许可](../../community/licenses.md)下发布的。

对于 Synopsys GUN Toolchain，它是在 [GPLv3 许可](https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/blob/arc-releases/COPYING)下发布的。

TensorFlow 库代码和第三方代码包含他们自己的许可证，在 [TFLite-Micro](https://github.com/tensorflow/tflite-micro) 中有说明。
