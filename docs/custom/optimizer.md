# Optimizer Configuration

Optimizer configuration files are used to define the optimizer, optimizer wrapper, and learning rate adjustment strategies during the model training process. In SSCMA, based on MMEngine's optimizer wrapper, it is a high-level abstraction that adds functionality to the native PyTorch optimizers and provides a unified interface. It supports various training strategies, such as mixed precision training, gradient accumulation, and gradient clipping, and defines the parameter update process, making it convenient to switch between different training strategies.

Optimizer configuration typically includes the following modules:

1. **Optimizer**: The optimizer is an algorithm used to update model parameters to minimize the loss function. Common optimizers include SGD, Adam, AdamW, etc.
2. **Optimizer Wrapper**: Defines the optimizer wrapper type and configures parameter adjustment strategies and other advanced features.
3. **Optimizer Parameter Adjustment Strategy**: Defines the strategy for dynamically adjusting optimizer parameters, such as linear learning rate adjustment, cosine annealing learning rate adjustment, etc.

Below is an example of the optimizer configuration for the RTMDet model in SSCMA, which introduces the detailed content of the optimizer configuration.

:::tip
For more information on optimizers, please refer to [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html) and [MMEngine - Optimizers](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optimizer.html). If you encounter issues accessing these links, it may be related to the validity of the web page links or network connectivity. Please check the legitimacy of the web page links and try again if necessary.
:::

## Optimizer

The optimizer module is used to define the type and parameters of the optimizer.

```python
optimizer=dict(
    type=AdamW,  # Type of optimizer
    lr=base_lr,  # Learning rate
    weight_decay=0.05,  # Weight decay
)
```

## Optimizer Wrapper

The optimizer wrapper module is used to further configure the optimizer's parameter adjustment strategies and other advanced features.

```python
optim_wrapper=dict(
    type=OptimWrapper,  # Type of optimizer wrapper
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.05),  # Optimizer configuration
    paramwise_cfg=dict(
        norm_decay_mult=0,  # Weight decay multiplier for normalization layers
        bias_decay_mult=0,  # Weight decay multiplier for bias terms
        bypass_duplicate=True,  # Whether to bypass duplicate parameters
    ),
)
```

## Optimizer Parameter Adjustment Strategy

The optimizer parameter adjustment strategy module is used to dynamically adjust the parameters of the optimizer.

```python
param_scheduler=[
    dict(
        type=LinearLR,  # Linear learning rate adjustment
        start_factor=1.0e-5,  # Starting factor
        by_epoch=False,  # Whether to adjust by epoch
        begin=0,  # Iteration to start adjustment
        end=1000,  # Iteration to end adjustment
    ),
    dict(
        type=CosineAnnealingLR,  # Cosine annealing learning rate adjustment
        eta_min=base_lr * 0.05,  # Minimum learning rate
        begin=max_epochs // 2,  # Epoch to start adjustment
        end=max_epochs,  # Epoch to end adjustment
        T_max=max_epochs // 2,  # Cosine cycle
        by_epoch=True,  # Whether to adjust by epoch
        convert_to_iter_based=True,  # Whether to convert to iteration-based adjustment
    ),
]
```
