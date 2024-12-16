# VAE Model Training

This section will introduce how to train the VAE (Variational Autoencoder) model for anomaly detection tasks. The VAE model is a type of generative model that can learn the distribution of data and generate data similar to the original data. In anomaly detection tasks, we can use the VAE model to learn the distribution of normal data and then calculate the anomaly score to determine whether the data is abnormal.

## Data Preparation

Before using the VAE model for anomaly detection, manual data collection is required. Here, we take the three-axis gyroscope as an example, and use the ESP32 for data collection.

### Compile Firmware

First, you need to compile the ESP32 firmware. You can refer to [ESP32 Compile Firmware](../../hardware/esp32/compile_firmware.md).

We provide the source code for data collection, which you can access at [SSCMA Example ESP32 - GADNN](https://github.com/Seeed-Studio/sscma-example-esp32/blob/dev/examples/gyro_anomaly_detection_nn/main/app_main.cpp) to get the source code.

Next, check the example program, you can see that the example defaults to using the QMA7981 three-axis gyroscope as the sensor, with a sampling frequency of 100Hz. Modify the macro `GYRO_SAMPLE_MODE` to 1 to enable data collection mode, and the collected data will be sent from the ESP32 to the PC via the serial port.

### Data Collection

Connect the ESP32 to the PC via a USB cable. Assuming the device is mounted at `/dev/ttyACM0` (Linux) or `COM3` (Windows), run the following command to collect data:

```sh
python3 tools/dataset_tool/read_serial.py \
    -sr 115200 \
    -p /dev/ttyACM0 \
    -f datasets/accel_3axis.csv
```

:::tip

When collecting data, ensure the device is placed steadily to avoid movement during the data collection process. If you need to detect anomalies when the device is moving, you can move the device during data collection, but ensure that the device's motion state is a normal motion state during the collection.

:::

### Data Preprocessing

After data collection is complete, we need to preprocess the data:

```sh
python tools/dataset_tool/signal_data_processing.py \
    -t train \
    -f datasets/accel_3axis.csv
```

After preprocessing is complete, the dataset will be saved in the `datasets/accel_3axis` directory.

## Model Training

Here, we take `vae_mirophone.py` as an example to show how to use SSCMA for VAE model training.

```sh
python3 tools/train.py \
    configs/anomaly/vae_mirophone.py \
    --cfg-options \
    data_root=$(pwd)/datasets/accel_3axis/
```

- `configs/anomaly/vae_mirophone.py`: Specifies the configuration file, defining the model and training settings.
- `--cfg-options`: Used to specify additional configuration options.
    - `data_root`: Sets the root directory of the dataset.

## Model Exporting and Verification

During the training process, you can view the training logs, export the model, and verify the model's performance at any time. Some metrics output during model verification are also displayed during the training process. Therefore, in this part, we will first introduce how to export the model and then discuss how to verify the accuracy of the exported model.

### Exporting the Model

Here, we take exporting the TFLite model as an example. You can use the following command to export TFLite models of different accuracies:

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
    configs/anomaly/vae_mirophone.py \
    work_dirs/epoch_100_int8.tflite \    
    --cfg-options \
    data_root=$(pwd)/datasets/accel_3axis/ 
```

### QAT

QAT (Quantization-Aware Training) is a method that simulates quantization operations during the model training process, allowing the model to gradually adapt to quantization errors, thereby maintaining high accuracy after quantization. SSCMA supports QAT, and you can refer to the following method to obtain a QAT model and verify it again.

```sh
python3 tools/quantization.py \
    configs/anomaly/vae_mirophone.py \
    work_dirs/epoch_100.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/accel_3axis/
```

After QAT training, the quantized model will be automatically exported, and its storage path will be `out/qat_model_test.tflite`. You can use the following command to verify it:

```sh
python3 tools/test.py \
    configs/anomaly/vae_mirophone.py \
    out/qat_model_test.tflite \    
    --cfg-options \
    data_root=$(pwd)/datasets/accel_3axis/ 
```


## Model Deployment

First, use the `xxd` tool to convert the TFLite model into a C array:

```sh
cp out/qat_model_test.tflite out/nn_model_tflite && \
xxd -i out/nn_model_tflite > out/nn_model.h
```

Then, copy the generated `nn_model.h` header file to the `sscma-example-esp32/examples/gyro_anomaly_detection_nn/main` directory of the ESP32 project, replacing the original `nn_model.h` file.

Finally, modify the macro `GYRO_SAMPLE_MODE` to 0, recompile the ESP32 project, and you can deploy the model to the ESP32.
