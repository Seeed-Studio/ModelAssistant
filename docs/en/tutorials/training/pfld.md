# Train the PFLD model
This section will show how to train PFLD model on meter datasets.
- [Train the PFLD model](#train-the-pfld-model)
    - [Dataset](#dataset)
    - [Config](#config)
    - [Train](#train)
        - [Description of all arguments](#description-of-all-arguments)
    - [Test & Eval](#test--eval)
        - [Test](#test)
        - [Eval](#eval)
    - [Reminders](#reminders)
    - [FAQs](#faqs)

## Dataset
We have already prepared a ready-to-use dataset.

- **Step 1.** Click [here](https://1drv.ms/u/s!AqG2uRmVUhlShtIhyd_7APHXEhpeXg?e=WwGx5m) to download the dataset.

- **Step 2.** Unzip the downloaded dataset and remember this file path, which will be used when modifying the configuration file.

## Config
Here we will choose the profile according to the task that we want to implement. We have prepared preconfigured files inside the the [configs](../../../../configs/pfld) folder.

For our meter reading detection example, we will use [pfld_mv2n_112.py](../../../../configs/pfld/pfld_mv2n_112.py) config file. This file will be mainly used to configure the dataset for training including the dataset location.

## Train
Execute the following command inside the activated conda virtual environment terminal to start training an end-to-end meter reading detection model.

```sh
python tools/train.py \
    ${TYPE} \
    ${CONFIG_FILE} \
    --work-dir ${WORK-DIR} \
    --gpu-id ${GPU-ID} \
    --cfg-options ${CFG-OPTIONS} \
```

### Description of all arguments
- `${TYPE}` Type for training model, [`mmdet`, `mmcls`, `mmpose`], `mmpose` for pfld。
- `${CONFIG_FILE}` Configuration file for model(under the configs directory).
- `--work-dir` Directory to save the model checkpoints and logs for the current experiments.
- `gpu-id` Id of gpu to use(only applicable to non-distributed training).
- `--cfg-options` Override some settings in the used config file, the key-value pair in xxx=yyy format will be merged into config file.

**Note:** Some parameters will be needed in `--cfg-options`, you can view this [website](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html) for more details, and change it with command like this:

```sh
--cfg-options \
    data_root=${DATA-ROOT} \
    load_from=${LOAD-FROM} \
```
- `${DATA-ROOT}` Path for the trainning dataset.
- `${LOAD-FROM}` Load models as a pre-trained model from a given path. This will not resume training.

After the training is completed, a model weight file will be generated under **~/Edgelab/work_dir/pfld_mv2n_112/exp1/latest.pth**. Remember the path to this file, which will be used when exporting the model.

## Test & Eval

### Test
Use this command to test your model:
```sh
python tools/test.py \
    mmpose \
    configs/pfld/pfld_mv2n_112.py \
    pfld_mv2n_112.pth \
    --no_show \
```

### Eval
Export other format model for evaluation, please read [pytorch2onnx](../export/pytorch2onnx.md) and [pytorch2tflite](../export/pytorch2tflite.md).

## Reminders
- 

## FAQs
- 