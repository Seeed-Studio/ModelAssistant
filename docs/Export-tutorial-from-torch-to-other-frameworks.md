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
python ./tool/export.py $TYPE $CONFIG --weights $WEIGHTS_PATH --data $REPRESENTATIVE_DATASET --tflite_type $TFLITE_TYPE --shape $INPUT_SHAPE --audio $AUDIO
```
##### Parameters description
- `$TYPE` Type for training model，['mmdet', 'mmcls', 'mmpose']。
- `$CONFIG` Configuration file for model(under the configs directory).
- `$WEIGHTS_PATH` Path of torch model.
- `$REPRESENTATIVE_DATASET` Path to representative dataset, it is recommended to use the training dataset, only for `int8`.
- `TFLITE_TYPE` Quantization type for tflite, `int8`, `fp16`, `fp32`, default: `int8`.
- `$INPUT_SHAPE` Shape of input, default: pfld model: '112' or '112 112', audio model: '8192'.
- `AUDIO` Choose audio dataset load code if given.

## Example
### pfld model
- Converting the pfld model (pfld.pth) from torch to tflite of int8,
the representative dataset (pfld_data) is located in the root 
directory, the weights of torch model is also located in the root directory, 
and the input image size is set to 112.

**Note：** TFLite model is saved in the same path as torch model. If you want to export the tflite model of fp16 or fp32,
you need to add the `--tflite_type` parameter. The audio model needs to add the `--audio` parameter.

#### Command
```shell
python ./tools/export.py mmpose configs/pfld/pfld_mv2n_112.py --weights pfld.pth --data pfld_data --shape 112
```

If the export is successful, the corresponding tflite save path will be displayed.

### fomo model
#### Command
```shell
python ./tools/export.py mmdet configs/fomo/fomo_mobnetv2_x8_voc.py --weights ./fomo.pth --data fomo_data --shape 92
```
