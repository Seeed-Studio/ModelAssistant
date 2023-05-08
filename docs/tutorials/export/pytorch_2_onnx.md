# PyTorch to ONNX (Experimental)

This chapter will describe how to convert and export PyTorch models to ONNX models.


## Preparation

### Environment Configuration

As the [Training](../training/overview.md) step, we recommend you to do it in a **virtual environment** during the model exporting phase. In the `edgelab` virtual environment, make sure that the [Installation - Prerequisites - Install Extra Dependencies](../../introduction/installation#step-4-install-extra-dependencies-optional) step has been completed.

::: tip

If you have configured a virtual environment but not activated it, you can activate it with the following command.

```sh
conda activate edgelab
```

:::

### Models and Weights

You also need to prepare the PyTorch model and its weights before exporting the model. For the model, you can find it in the [Config](../config.md) section, we have already preconfigured. For the weights, you can refer to the following steps to get the model weights.

- Refer to [Training](../training/overview.md) section and choose a model, and train to get the model weights.

- Or download the EdgeLab official pre-trained weights from our [GitHub Releases - Model Zoo](https://github.com/Seeed-Studio/EdgeLab/releases/tag/model_zoo).


## Model Transform

For model transformation (convert and export), the relevant commands with some common parameters are listed.

```sh
python3 tools/torch2onnx.py \
    <TASK> \
    <CONFIG_FILE_PATH> \
    --checkpoint <CHECKPOINT_FILE_PATH> \
    --simplify <SIMPLIFY> \
    --shape <SHAPE>
```

### Transform Parameters

You need to replace the above parameters according to the actual scenario, the details of parameters are described as follows.

- `<TASK>` - Type of model, choose from: `['det', 'cls', 'pose']`

- `<CONFIG_FILE_PATH>` - Path to the model config file

- `<CHECKPOINT_FILE_PATH>` - Path to the model weights file

- `<SIMPLIFY>` - Whether to simplify the model, default `False`

- `<SHAPE>` - The dimensionality of the model's input tensor, default `112`

::: tip

For more parameters supported, please refer to the source code `tools/torch2onnx.py`.

:::

### Transform Examples

Here are some model conversion examples for reference.

::: code-group

```sh [FOMO Model Conversion]
python3 tools/torch2onnx.py \
    det \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    --checkpoint "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint)" \
    --shape 96
```

```sh [PFLD Model Conversion]
python3 tools/torch2onnx.py \
    pose \
    configs/pfld/pfld_mv2n_112.py \
    --checkpoint "$(cat work_dirs/pfld_mv2n_112/last_checkpoint)" \
    --shape 112
```

:::


## Model Validation

Since in the process of exporting the model, EdgeLab will do some optimization for the model using some tools, such as model pruning, distillation, etc. Although we have tested and evaluated the model weights during the training process, we recommend you to validate the exported model again.

```sh
python3 tools/test.py \
    <TASK> \
    <CONFIG_FILE_PATH> \
    <CHECKPOINT_FILE_PATH> \
    --out <OUT_FILE_PATH> \
    --work-dir <WORK_DIR_PATH> \
    --cfg-options <CFG_OPTIONS>
```

### Validation Parameters

You need to replace the above parameters according to the actual scenario, and descriptions of different parameters are as follows.

- `<TASK>` - Type of model, choose from: `['det', 'cls', 'pose']`

- `<CONFIG_FILE_PATH>` - Path to the model configuration file

- `<CHECKPOINT_FILE_PATH>` - Path to the model weights file

- `<OUT_FILE_PATH>` - (Optional) Path to the output folder for the validation results

- `<WORK_DIR_PATH>` - (Optional) Path to the working directory

- `<CFG_OPTIONS>` - (Optional) Configuration file parameter override, please refer to [Config - Parameterized Configuration](../config.md#parameterized-configuration)

::: tip

For more parameters supported, please refer to the source code `tools/test.py`.

:::

### Validation Example

::: code-group

```sh [FOMO Model Validation]
python3 tools/test.py \
    det \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint | sed -e 's/.pth/.onnx/g')" \
    --cfg-options \
        data_root='datasets/mask'
```

```sh [PFLD Model Validation]
python3 tools/test.py \
    pose \
    configs/pfld/pfld_mv2n_112.py \
    "$(cat work_dirs/pfld_mv2n_112/last_checkpoint | sed -e 's/.pth/.onnx/g')" \
    --cfg-options \
        data_root='datasets/meter'
```

:::
