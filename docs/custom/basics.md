# Basic Configuration Structure

The configuration files used by SSCMA are located in the `configs` directory, which are used for training different models under different tasks. We have divided subfolders according to different tasks, and multiple model training pipeline parameters are saved in each subfolder.

:::tip

The task folder named `_base_` is the inheritance object for our other tasks. For detailed instructions on configuration file inheritance, please refer to [MMEngine - Configuration File Inheritance](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html#id3).

:::

We use Python dictionaries and lists to define parameters for models, data loading, training, and evaluation. Below is the structure of the configuration file, the role of each part, and the parameters that generally need to be adjusted.

## Import Modules

Before starting to build your configuration file, you need to import the necessary modules as shown below:

```python
import torch.nn as nn
from mmengine.registry import OPTIMIZERS
```

## Model Configuration

- `num_classes`: The number of categories, for mask detection, it is usually 2 (wearing a mask and not wearing a mask).
- `widen_factor`: The model width factor, used to adjust the width of the model.

## Data Configuration

- `dataset_type`: Specifies the type of dataset.
- `data_root`: The root directory of the dataset.
- `train_ann`, `train_data`, `val_ann`, `val_data`: The annotation files and data directories for training and validation data.
- `height`, `width`, `imgsz`: The size of the input image.

## Training Configuration

- `batch`, `workers`, `persistent_workers`: The batch size, number of worker threads, and persistent worker threads during training.
- `val_batch`, `val_workers`: The batch size and number of worker threads during validation.
- `lr`, `epochs`, `weight_decay`, `momentum`: Learning rate, training periods, weight decay, and momentum.

## Hooks

- `default_hooks`: Defines hooks during the training process, such as visualization hooks.
- `visualizer`: Defines the visualizer.

## Data Preprocessing

- `data_preprocessor`: Defines the parameters for data preprocessing, such as mean, standard deviation, and color conversion.

## Model Structure

Defines the type of model, data preprocessor, backbone network, and head network configuration.

## Deployment Configuration

- `deploy`: Defines the data preprocessor configuration when the model is deployed.

## Image Decoding Backend

- `imdecode_backend`: Specifies the backend for image decoding.

## Preprocessing Pipeline

- `pre_transform`, `train_pipeline`, `test_pipeline`: Defines the preprocessing pipeline for training and testing data.

## Data Loaders

- `train_dataloader`, `val_dataloader`, `test_dataloader`: Defines the configuration for training, validation, and testing data loaders.

## Optimizer Configuration

- `optim_wrapper`: Defines the type and parameters of the optimizer.

## Evaluators

- `val_evaluator`, `test_evaluator`: Defines the evaluators for validation and testing.

## Training Configuration

- `train_cfg`: Defines the training configuration, such as whether to train by periods and the maximum number of periods.

## Learning Strategy

- `param_scheduler`: Defines the strategy of the learning rate scheduler.

The configuration file covers all aspects from data preprocessing to model training and evaluation. According to specific training needs, parameters that may need to be adjusted include learning rate, batch size, training periods, optimizer parameters, data augmentation strategies, etc. The adjustment of these parameters will directly affect the model's performance and training results.
