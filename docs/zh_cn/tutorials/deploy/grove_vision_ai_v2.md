# 在 Grove Vision AI V2 上部署 SSCMA 模型

本示例为 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 包含的模型在 Grove Vision AI V2 模块的部署教程，

在 Grove Vision AI V2 上部署 SSCMA 模型，您需要先将模型转换为量化的 TensorFlow Lite 格式，Vela 转换器转换后，您可以将模型部署到 Grove Vision AI V2 模块上。SSCMA 在导出工具中添加了这个功能，您可以在导出过程中，添加参数以 `--format vela` 来导出 Vela 模型。成功导出后，剩下的步骤就和 [SSCMA - 模型部署](overview)中一致了。

此外，您也可以尝试手动构建固件来适配模型。

## 在 Linux 环境下构建固件

以下步骤已在 Ubuntu 20.04 PC 上测试通过。

### 安装依赖

```bash
sudo apt install make
```

### 下载 Arm GNU 工具链

```bash
cd ~
wget https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
```

### 解压文件

```bash
tar -xvf arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
```

### 添加到 PATH

```bash
export PATH="$HOME/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi/bin/:$PATH"
```

### 克隆以下仓库并进入 Seeed_Grove_Vision_AI_Module_V2 文件夹

```bash
git clone --recursive https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2.git
cd Seeed_Grove_Vision_AI_Module_V2
```

### 编译固件

```bash
cd EPII_CM55M_APP_S
make clean
make
```

输出的 ELF 文件位于 `/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf`。

### 生成固件镜像文件

```bash
cd ../we2_image_gen_local/
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/
./we2_local_image_gen project_case1_blp_wlcsp.json
```

输出的固件镜像位于 `./output_case1_sec_wlcsp/output.img`。

### 刷入固件

您可以使用 SSCMA Web Toolkit 或者 Grove Vision AI V2 的 USB 转串口工具刷入固件，也可以直接使用 Xmodem 协议刷入固件。
