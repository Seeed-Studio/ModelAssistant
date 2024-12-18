# Deploying SSCMA Models on Grove Vision AI V2

This example is a deployment tutorial for models included in [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) on the Grove Vision AI V2 module.

To deploy SSCMA models on the Grove Vision AI V2, you need to first convert the models into quantized TensorFlow Lite format. After conversion with the Vela converter, you can deploy the models to the Grove Vision AI V2 module. SSCMA has added this feature in the export tool, allowing you to add the parameter `--format vela` during the export process to export Vela models. Once successfully exported, the remaining steps are consistent with [SSCMA - Model Deployment](overview).

Additionally, you can attempt to manually build the firmware to accommodate the model.

## Building Firmware in Linux Environment

The following steps have been tested on Ubuntu 20.04 PC.

### Install Dependencies

```bash
sudo apt install make
```

### Download Arm GNU Toolchain

```bash
cd ~
wget https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz 
```

### Extract the File

```bash
tar -xvf arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
```

### Add to PATH

```bash
export PATH="$HOME/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi/bin/:$PATH"
```

### Clone the Following Repository and Enter the Seeed_Grove_Vision_AI_Module_V2 Folder

```bash
git clone --recursive https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2.git 
cd Seeed_Grove_Vision_AI_Module_V2
```

### Compile the Firmware

```bash
cd EPII_CM55M_APP_S
make clean
make
```

The output ELF file is located at `/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf`.

### Generate Firmware Image File

```bash
cd ../we2_image_gen_local/
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/
./we2_local_image_gen project_case1_blp_wlcsp.json
```

The output firmware image is located at `./output_case1_sec_wlcsp/output.img`.

### Flashing the Firmware

You can use the SSCMA Web Toolkit or Grove Vision AI V2's USB-to-serial tool to flash the firmware, or you can directly use the Xmodem protocol to flash the firmware.
