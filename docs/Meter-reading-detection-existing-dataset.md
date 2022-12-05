# Train a meter reading detection model with existing dataset

**Note:** Before proceeding, please follow the steps under **Configure host environment** inside the [README.md](https://github.com/Seeed-Studio/Edgelab/blob/master/README.md)

### Prepare dataset

We have already prepared a ready-to-use dataset

- **Step 1.** Click [here](https://1drv.ms/u/s!AqG2uRmVUhlShtIhyd_7APHXEhpeXg?e=WwGx5m) to download the dataset

- **Step 2.** Unzip the downloaded dataset and remember this file path, which will be used when modifying the configuration file

### Configure the profile

Here we will choose the profile according to the task that we want to implement. We have prepared preconfigured files inside the the [configs](https://github.com/Seeed-Studio/edgelab/tree/master/configs) folder.

For our meter reading detection example, we will use [pfld_mv2n_112.py](https://github.com/Seeed-Studio/Edgelab/blob/master/configs/pfld/pfld_mv2n_112.py) config file. This file will be mainly used to configure the dataset for training including the dataset location.

### Start training 

Execute the following command inside the activated conda virtual environment terminal to start training an end-to-end meter reading detection model.

```sh
python tools/train.py mmpose configs/pfld/pfld_mv2n_112.py
```

The format of the above command looks like below

```sh
python tools/train.py <task_type> <config_file_location> --gpus=<cpu_or_gpu> --cfg-options runner.max_epochs=<number_of_epochs>
```

where:

- <task_type> refers to either **mmcls** for classfication, **mmdet** for detection and **mmpose** for pose estimation
- <config_file_location> refers to the path where the model configuration is located 
- <cpu_or_gpu> refers to specifying whether you want to train on CPU or GPU. Type **0** CPU and **1** for GPU
- --cfg-options runner.max_epochs=<number_of_epochs> refers to the number of training cycles

After the training is completed, a model weight file will be generated under  **~/edgelab/work_dir/pfld_mv2n_112/exp1/latest.pth**. Remember the path to this file, which will be used when exporting the model.

### Export to ONNX 

After the model training is completed, you can export the **.pth file** to the **ONNX file** format and convert it to other formats you want to use through ONNX. Assuming that the environment is in this project path, you can export the object detection model you just trained to the ONNX format by running the following command:

```sh
python tools/torch2onnx.py mmpose --config configs/pfld/pfld_mv2n_112.py --checkpoint ./work_dir/pfld_mv2n_112/exp
1/latest.pth --shape 112
```

where:

- --config refers to the model configuration file
- --checkpoint refers to the weight file after model training
- --shape refer to the size of the model input data