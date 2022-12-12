# Export tutorial from torch model to other frameworks

**Note:** Before proceeding, please follow the steps under **Configure host environment** inside the [README.md](https://github.com/Seeed-Studio/Edgelab/blob/master/README.md)

## 1. Convert torch to TFLite

Quantize the weight of the model trained by torch from float32 to int8, thereby reducing memory and computing power requirements, 
so the model can be applied to low-power embedded devices. The current mainstream 
quantization method is TFLite, we provides the conversion method flow as below.

### Preparation
1. Make sure the torch model is ready.
2. Converting TFLite requires a representative dataset, please prepare a standard 
dataset (100 data) similar to the training data, or use the provided [Representative dataset](https://1drv.ms/u/s!AgNatz-E2yLkhEN9Xh9bsuSu9e7G?e=nyKFZ0). 
It is important to ensure that the representative dataset used is similar to the training data.

#### Python command
```shell
python .\tool\export.py --weights $WEIGHTS_PATH --data_root $REPRESENTATIVE_DATASET --name $MODEL_TYPE --shape $INPUT_SHAPE
```
##### Parameters description
- `$WEIGHTS_PATH` Path of torch model
- `$REPRESENTATIVE_DATASET` Path to representative dataset
- `$MODEL_TYPE` Type of model needs to be converted.
- `$INPUT_SHAPE` Shape of input

### Example
- Converting the pfld model (pfld.pth) from torch to tflite, 
the representative dataset (pfld_data) is located in the root 
directory, the weights of torch model is also located in the root directory, 
and the input image size is set to 112.

**Note：** TFLite model is saved in same path as torch model.

### Command
```shell
python .\tool\export.py --weights pfld.pth --data_root pfld_data --name pfld --shape 112
```

If the export is successful, the corresponding tflite save path will be displayed.

