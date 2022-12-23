# Train and deploy an analog meter reading detection model with existing dataset

This guide will explain how you can train an analog meter detection model using an existing dataset and deploy on the Grove - Vision AI Module. This will only be accurate for the dataset that we provide. However, if you want to generate a more accurate model with your own dataset for your own application, please follow [this tutorial](https://github.com/Seeed-Studio/Edgelab/blob/master/docs/Meter-reading-detection-own-dataset.md).

### Prerequisites

Please prepare the following

- Ubuntu PC with an internet connection
- [Grove - Vision AI Module](https://www.seeedstudio.com/Grove-Vision-AI-Module-p-5457.html)
- USB Type-C cable

### Configure host environment 

We first need to configure the host PC. Here we have used a PC with Ubuntu 22.04 installed.

- **Step 1.** Clone the following repo and access it

```
https://github.com/Seeed-Studio/edgelab
cd edgelab
```

- **Step 2.** Configure the host environment by running the following script which will download and install the relevant dependencies 

```sh
python3 tools/env_config.py
```

The above script will mainly install **PyTorch, MMCV, MMClassification, MMDetection and MMPose**.

**Note:** The above script will add various environment variables to the file ~/.bashrc, establish a conda virtual environment called edgelab, install relevant dependencies inside the virual environment. 

- **Step 3.** Eventhough everything is initialized at this point, they are not activated. Activate conda, virtual environments, and other related environment variables

```sh
source ~/.bashrc
conda activate edgelab
```

Now we have finished configuring the host environment 

### Generate firmware image

We need to generate a firmware image for the Grove - Vision AI to support the analog meter reading detection model because the default firmware which is pre-installed out-of-the-box does not support it. 

- **Step 1:** Download GNU Development Toolkit

```sh
cd ~
wget https://github.com/foss-for-synopsys-dwc-arc-processors/toolchain/releases/download/arc-2020.09-release/arc_gnu_2020.09_prebuilt_elf32_le_linux_install.tar.gz
```

- **Step 2:** Extract the file

```sh
tar -xvf arc_gnu_2020.09_prebuilt_elf32_le_linux_install.tar.gz
```

- **Step 3:** Add **arc_gnu_2020.09_prebuilt_elf32_le_linux_install/bin** to **PATH**

```sh
export PATH="$HOME/arc_gnu_2020.09_prebuilt_elf32_le_linux_install/bin:$PATH"
```

- **Step 4:** Clone the following repository and go into the directory

```sh
git clone https://github.com/Seeed-Studio/Edgelab
cd Edgelab/examples/grove_ai
```

- **Step 5:** Download related third party, tflite model and library data (only need to download once)

```sh
make download
```

- **Step 6:** Compile the firmware

```sh
make APP=meter
make flash
```

This will generate **output.img** inside **tools/image_gen_cstm/output/** directory


- **Step 7:** Generate firmware image **firmware.uf2** file so that we can later flash it directly to the Grove - Vision AI Module

```sh
python3 tools/ufconv/uf2conv.py -t 0 -c tools/image_gen_cstm/output/output.img -o firmware.uf2
```

### Download pre-trained model 

To get started and experience the analog meter detection in the fastest way, we have prepared a pre-trained model for you to use. If you use this model, you can directly jump to the section "Convert TFLite to UF2". Otherwise, follow the next steps.

```sh
wget https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/pfld_meter_int8.tflite
```

The above will download the model file as **pfld_meter_int8.tflite**

### Prepare dataset

We have already prepared a ready-to-use dataset

- **Step 1.** Click [here](https://1drv.ms/u/s!AqG2uRmVUhlShtIhyd_7APHXEhpeXg?e=WwGx5m) to download the dataset

- **Step 2.** Unzip the downloaded dataset and remember this file path, which will be used when passing parameters to the training command later

### Configuration file

Here we will choose the profile according to the task that we want to implement. We have prepared preconfigured files inside the the [configs](https://github.com/Seeed-Studio/edgelab/tree/master/configs) folder.

For our meter reading detection example, we will use [pfld_mv2n_112.py](https://github.com/Seeed-Studio/Edgelab/blob/master/configs/pfld/pfld_mv2n_112.py) config file. This file will be mainly used to configure the dataset for training including the dataset location.

### Start training 

Execute the following command inside the activated conda virtual environment terminal to start training an end-to-end analog meter reading detection model.

```sh
python tools/train.py mmpose configs/pfld/pfld_mv2n_112.py --gpus=1 --cfg-options runner.max_epochs=100 data.train.index_file=/meter/train/annotations data.train.img_dir=/meter/train/images data.val.index_file=/meter/val/annotations data.val.img_dir=/meter/val/images data.test.index_file=/meter/test/annotations data.test.img_dir=/meter/test/images
```

The format of the above command looks like below

```sh
python tools/train.py <task_type> <config_file_location> --gpus=<cpu_or_gpu> --cfg-options runner.max_epochs=<number_of_epochs> data.train.index_file=<absolute_path_to_annotations_in_train> data.train.img_dir=<absolute_path_to_images_in_train> data.val.index_file=<absolute_path_to_annotations_in_val> data.val.img_dir=<absolute_path_to_images_in_val> data.test.index_file=<absolute_path_to_annotations_in_test> data.test.img_dir=<absolute_path_to_images_in_test>
```

where:

- <task_type> refers to either **mmcls** for classfication, **mmdet** for detection and **mmpose** for pose estimation
- <config_file_location> refers to the path where the model configuration is located 
- <cpu_or_gpu> refers to specifying whether you want to train on CPU or GPU. Type **0** CPU and **1** for GPU
- --cfg-options runner.max_epochs=<number_of_epochs> refers to the number of training cycles
- --cfg-options data.train.index_file=<absolute_path_to_annotations_in_train> refers to the location of the annotations files under train set
- --cfg-options data.train.img_dir=<absolute_path_to_images_in_train> refers to the location of the images files under train set
- --cfg-options data.val.index_file=<absolute_path_to_annotations_in_val> refers to the location of the annotations files under validation set
- --cfg-options data.val.index_file=<absolute_path_to_annotations_in_val> refers to the location of the image files under validation set
- --cfg-options data.test.index_file=<absolute_path_to_annotations_in_test> refers to the location of the annotations files under test set
- --cfg-options data.test.img_dir=<absolute_path_to_images_in_test> refers to the location of the image files under test set


After the training is completed, a model weight file will be generated under  **~/edgelab/work_dir/pfld_mv2n_112/exp1/best_loss_epoch.pth**. Remember the path to this file, which will be used when exporting the model.

### Export PyTorch to TFLite

After the model training is completed, you can export the **.pth file** to the **TFLite file** format. This is important because TFLite format is more optimized to run on low power hardware. Assuming that the environment is in this project path, you can export the meter reading detection model you just trained to the TFLite format by running the following command:

```sh
python tools/export.py configs/pfld/pfld_mv2n_112.py --weights work_dir/pfld_mv2n_112/exp1/best_loss_epoch.pth --data_root /meter/train/images
```

The format of the above command looks like below

```sh
python tools/export.py configs/pfld/pfld_mv2n_112.py --weights <location_to_pth_from_training> --data_root <location_to_images_directory_of__train_or_val>
```

where:

- --weights refers to the the .pth file that was generated during training
- --data_root refers to the images directory of either train or val

This will generate a **best_loss_epoch_int8.tflite** file 

### Convert TFLite to UF2

Now we will convert the generated TFLite file to a UF2 file so that we can directly flash the UF2 file into Grove - Vision AI Module

```sh
python3 examples/vision_ai/tools/ufconv/uf2conv.py -f GROVEAI -t 1 -c ~/Edgelab/work_dirs/pfld_mv2n_112/exp1/best_loss_epoch_int8.tflite -o model.uf2
```
This will generate a **model.uf2** file 

Alternatively, you can convert the pre-trained pfld_meter_int8.tflite from before to UF2 as well

```sh
python3 examples/vision_ai/tools/ufconv/uf2conv.py -f GROVEAI -t 1 -c ~/pfld_meter_int8.tflite -o model.uf2
```

### Flash firmware and model

This explains how you can flash the previously generated firmware and the model file to Grove - Vision AI Module.

- **Step 1:** Connect Grove - Vision AI Module to the host PC via USB Type-C cable 

<div align=center><img width=460 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/47.png"/></div>

- **Step 2:** Double-click the boot button on Grove - Vision AI Module to enter mass storage mode

<div align=center><img width=220 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/48.png"/></div>

- **Step 3:** After this you will see a new storage drive shown on your file explorer as **GROVEAI**

<div align=center><img width=250 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/19.jpg"/></div>

- **Step 4:** Drag and drop the previous **firmware.uf2** at first, and then the **model.uf2** file to GROVEAI drive

Once the copying is finished **GROVEAI** drive will disapper. This is how we can check whether the copying is successful or not.

## View live detection results

- **Step 1:** After loading the firmware and connecting to PC, visit [this URL](https://files.seeedstudio.com/grove_ai_vision/index.html)

- **Step 2:** Click **Connect** button. Then you will see a pop up on the browser. Select **Grove AI - Paired** and click **Connect**
  
<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/13.jpg"/></div>

<div align=center><img width=400 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/12.png"/></div>

Upon successful connection, you will see a live preview from the camera. Here the camera is pointed at an analog meter.

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/14.png"/></div>

Now we need to set 3 points which is the center point, start point and the end point. 

- **Step 2:** Click on **Set Center Point** and click on the center of the meter. you will see a pop up confirm the location and press **OK**

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/15.png"/></div>

You will see the center point is already recorded

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/16.png"/></div>

- **Step 2:** Click on **Set Start Point** and click on the first indicator point. you will see a pop up confirm the location and press **OK**

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/17.png"/></div>

You will see the first indicator point is already recorded

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/18.png"/></div>

- **Step 3:** Click on **Set End Point** and click on the last indicator point. you will see a pop up confirm the location and press **OK**

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/19.png"/></div>

You will see the last indicator point is already recorded

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/20.png"/></div>

- **Step 4:** Set the measuring range according to the first digit and last digit of the meter. For example, he we set as **From:0 To 0.16**

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/21.png"/></div>

- **Step 5:** Set the number of decimal places that you want the result to display. Here we set as 2

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/22.png"/></div>

Finally you can see the live meter reading results as follows

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/Edgelab/meter-own-github/meter.gif"/></div>