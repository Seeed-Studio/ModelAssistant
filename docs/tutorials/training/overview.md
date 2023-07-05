# Model Training

EdgeLab currently supports the following models. You can refer to the corresponding tutorials to complete the training of the models and obtain the model weights.

- [FOMO Model](./fomo.md): Face mask detection

- [PFLD Model](./pfld.md): Pointer meter reading

- [YOLOv5 Model](./yolov5.md): digital meter reading

::: tip

Before start training, we recommend you to read [Config](../config.md) and [Datasets](../datasets.md) sections first.

:::

## Parameter Descriptions

For more parameters during model training, you can refer the code below.

```sh
python3 tools/train.py --help
# positional arguments:
#   config                the model config file path
# optional arguments:
#   -h, --help            show this help message and exit
#   --work_dir WORK_DIR   the directory to save logs and models
#   --amp                 enable automatic-mixed-precision during training (https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
#   --auto_scale_lr       enable automatic-scale-LR during training
#   --resume [RESUME]     resume training from the checkpoint of the last epoch (or a specified checkpoint path)
#   --no_validate         disable checkpoint evaluation during training
#   --no_persistent_workers
#                         disable persistent workers for dataloaders
#   --device DEVICE       the device used for convert & export
#   --launcher {none,pytorch,slurm,mpi}
#                         the job launcher for MMEngine
#   --cfg_options CFG_OPTIONS [CFG_OPTIONS ...]
#                         override some settings in the used config, the key-value pair in 'xxx=yyy' format will be merged into config file
#   --local_rank LOCAL_RANK
#                         set local-rank for PyTorch
#   --dynamo_cache_size DYNAMO_CACHE_SIZE
#                         set dynamo-cache-size limit for PyTorch
#   --input_shape INPUT_SHAPE [INPUT_SHAPE ...]
#                         Extension: input data shape for model parameters estimation, e.g. 1 3 224 224
```
