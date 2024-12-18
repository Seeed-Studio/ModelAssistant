# PFLD Model Training

This section will introduce how to train the PFLD meter model on the PFLD meter dataset. The PFLD model is proposed in the paper [PFLD: A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf).

## Dataset Preparation

Before training the PFLD model, we need to prepare the dataset. Here, we take the already annotated Meter dataset as an example, which you can download from [SSCMA - Public Datasets](../../datasets/public#obtaining-public-datasets).

## Model Selection and Training

SSCMA offers various PFLD model configurations, and you can choose the appropriate model for training based on your needs.

```sh
pfld_mbv2_1000e.py
pfld_mbv3l_192_1000e.py
```

Here, we take `pfld_mbv2_1000e.py` as an example to show how to use SSCMA for PFLD model training.

```sh
python3 tools/train.py \
    configs/pfld/pfld_mbv2_1000e.py \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    epochs=100 \
    val_workers=2
```

- `configs/pfld/pfld_mbv2_1000e.py`: Specifies the configuration file, defining the model and training settings.
- `--cfg-options`: Used to specify additional configuration options.
    - `data_root`: Sets the root directory of the dataset.
    - `epochs`: Sets the maximum number of training epochs.

After the training is complete, you can find the trained model in the `work_dirs/pfld_mbv2_1000e` directory.

:::details

```sh
12/16 06:40:25 - mmengine - INFO - Exp name: pfld_mbv2_1000e_20241216_062913
12/16 06:40:25 - mmengine - INFO - Saving checkpoint at 100 epochs
12/16 06:40:25 - mmengine - INFO - Epoch(val) [100][30/30]    keypoint/Acc: 0.8538  data_time: 0.0116  time: 0.0227
```

During the training process, you can view the training logs and metrics such as keypoint accuracy.

:::

:::tip

If your virtual environment is configured but not activated, you can activate the virtual environment with the following command:

```sh
conda activate sscma
```

:::

## Model Exporting and Verification

During the training process, you can view the training logs, export the model, and verify the model's performance at any time. Some metrics output during model verification are also displayed during the training process. Therefore, in this part, we will first introduce how to export the model and then discuss how to verify the accuracy of the exported model.

### Exporting the Model

Here, we take exporting the TFLite model as an example. You can use the following command to export TFLite models of different accuracies:

```sh
python3 tools/export.py \
    configs/pfld/pfld_mbv2_1000e.py \
    work_dirs/epoch_100.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    val_workers=2 \
    --imgsz 112 112 \
    --format tflite \
    --image_path $(pwd)/datasets/meter/val/images
```

:::warning

We recommend using the same resolution for training and exporting. In the current situation, using different resolutions for training and exporting may result in reduced model accuracy or complete loss of accuracy.

:::

:::tip

During the export process, an internet connection may be required to install certain dependencies. If you cannot access the internet, please ensure that the following dependencies are already installed in the current Python environment:

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

In addition, `onnx2tf` may also need to download calibration-related data during runtime. You can refer to the following link to download it in advance to the SSCMA root directory.

```sh
wget https://github.com/PINTO0309/onnx2tf/releases/download/1.20.4/calibration_image_sample_data_20x128x128x3_float32.npy  \
    -O calibration_image_sample_data_20x128x128x3_float32.npy
```

:::

### Verifying the Model

After exporting the model, you can use the following command to verify its performance:

```sh
python3 tools/test.py \
    configs/pfld/pfld_mbv2_1000e.py \
    work_dirs/epoch_100_int8.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    val_workers=2 
```

### QAT

QAT (Quantization-Aware Training) is a method that simulates quantization operations during the model training process, allowing the model to gradually adapt to quantization errors, thereby maintaining high accuracy after quantization. SSCMA supports QAT, and you can refer to the following method to obtain a QAT model and verify it again.

```sh
python3 tools/quantization.py \
    configs/pfld/pfld_mbv2_1000e.py \
    work_dirs/epoch_100.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    epochs=100 \
    val_workers=2
```

After QAT training, the quantized model will be automatically exported, and its storage path will be `out/qat_model_test.tflite`. You can use the following command to verify it:

```sh
python3 tools/test.py \
    configs/pfld/pfld_mbv2_1000e.py \
    out/qat_model_test.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    val_workers=2  
```
