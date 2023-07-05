# 模型训练

EdgeLab 目前支持以下模型，您可以参考对应的教程，完成模型的训练，获得模型权重。

- [FOMO 模型](./fomo.md): 口罩检测

- [PFLD 模型](./pfld.md): 指针表计读数

- [YOLOv5 模型](./yolov5.md): 数字表读数

::: tip

在开始训练前，我们建议您先阅读[模型配置](../config.md)和[数据集](../datasets.md)部分内容。

:::

### 参数说明

您需要将以下部分参数根据实际情况进行替换，各个不同参数的具体说明如下:

```sh
python3 tools/train.py --help

# Train EdgeLab models

# positional arguments:
#   config                the model config file path

# optional arguments:
#   -h, --help            show this help message and exit
#   --work_dir WORK_DIR, --work-dir WORK_DIR
#                         the directory to save logs and models
#   --amp                 enable automatic-mixed-precision during training (https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
#   --auto_scale_lr, --auto-scale-lr
#                         enable automatic-scale-LR during training
#   --resume [RESUME]     resume training from the checkpoint of the last epoch (or a specified checkpoint path)
#   --no_validate, --no-validate
#                         disable checkpoint evaluation during training
#   --no_persistent_workers, --no-persistent-workers
#                         disable persistent workers for dataloaders
#   --device DEVICE       the device used for convert & export
#   --launcher {none,pytorch,slurm,mpi}
#                         the job launcher for MMEngine
#   --cfg_options CFG_OPTIONS [CFG_OPTIONS ...], --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
#                         override some settings in the used config, the key-value pair in 'xxx=yyy' format will be merged into config file
#   --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
#                         set local-rank for PyTorch
#   --dynamo_cache_size DYNAMO_CACHE_SIZE, --dynamo-cache-size DYNAMO_CACHE_SIZE
#                         set dynamo-cache-size limit for PyTorch
#   --input_shape INPUT_SHAPE [INPUT_SHAPE ...], --input-shape INPUT_SHAPE [INPUT_SHAPE ...]
#                         Extension: input data shape for model parameters estimation, e.g. 1 3 224 224
```

### 部署

在导出模型后，你可以将模型部署到边缘计算设备上进行测试和评估。你可以参考 [examples](../../examples/examples.md) 部分来了解更多关于如何部署模型的信息。
