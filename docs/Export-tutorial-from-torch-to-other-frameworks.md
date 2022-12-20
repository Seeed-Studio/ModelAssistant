# Export tutorial from torch model to other frameworks

**Note:** Before proceeding, please follow the steps under **Configure host environment** inside the [README.md](https://github.com/Seeed-Studio/Edgelab/blob/master/README.md)

## 1. Convert torch to TFLite

Quantize the weight of the model trained by torch from float32 to int8, thereby reducing memory and computing power requirements, 
so the model can be applied to low-power embedded devices. The current mainstream 
quantization method is TFLite, we provides the conversion method flow for model supported in the repository as below.

### Preparation
1. Make sure the torch model is ready.
2. Converting TFLite requires a representative dataset, please use training dataset or prepare a standard 
dataset (100 data) similar to the training data, we recommend you to use the training dataset.
It is important to ensure that the representative dataset used is similar to the training data.

#### Python command
```shell
python ./tool/export.py $CONFIGS --weights $WEIGHTS_PATH --data_root $REPRESENTATIVE_DATASET --types $MODEL_TYPE --shape $INPUT_SHAPE --classes $AUDIO_CLASSES --fp16 $FP16
```
##### Parameters description
- `$CONFIGS` Configuration file for model(under the configs directory).
- `$WEIGHTS_PATH` Path of torch model.
- `$REPRESENTATIVE_DATASET` Path to representative dataset, it is recommended to use the training dataset.
- `$MODEL_TYPE` Type of model needs to be converted, 1 for 1d dataset of audio, 1 for 2d dataset for image, default: 2.
- `$INPUT_SHAPE` Shape of input, default: pfld model: '112' or '112 112', audio model: '8192'.
- `$AUDIO_CLASSES` Output numbers only for audio models, default: '4'.
- `FP16` Convert tflite model for fp16 quantization if this parameter is given, else int8 quantization.

### Example
- Converting the pfld model (pfld.pth) from torch to tflite, 
the representative dataset (pfld_data) is located in the root 
directory, the weights of torch model is also located in the root directory, 
and the input image size is set to 112.

**Note：** TFLite model is saved in same path as torch model. The representative dataset(pfld_data) and 
torch weights are both located in the root directory, input image size is 112 and get int8 model.

### Command
```shell
python ./tool/export.py configs/pfld/pfld_mv2n_112.py --weights pfld.pth --data_root pfld_data --shape 112
```

If the export is successful, the corresponding tflite save path will be displayed.

