# 优化器配置

优化器配置文件用于定义模型训练过程中的优化器、优化器封装和学习率调整策略。SSCMA 中基于 MMEngine 的优化器封装是一个高级抽象，在 Pytorch 原生优化器的基础上，增加了功能并提供了统一接口，可支持多种训练策略，如混合精度训练、梯度累加和梯度截断，并定义了参数更新流程，方便实现不同训练策略的切换。

优化器配置通常包含以下几个模块：

1. **优化器**：优化器是用于更新模型参数以最小化损失函数的算法。常见的优化器包括 SGD、Adam、AdamW 等。
2. **优化器封装**：定义优化器封装类型，并配置参数调整策略和其他高级功能。
3. **优化器参数调整策略**：定义动态调整优化器参数的策略，如线性学习率调整、余弦退火学习率调整等。

下面以 SSCMA 中 RTMDet 模型的优化器配置为例，介绍优化器配置的详细内容。

:::tip
关于优化器的更多信息，请参考 [PyTorch 优化器](https://pytorch.org/docs/stable/optim.html)、[MMEngine - 优化器](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optimizer.html)。
:::


## 优化器

优化器模块用于定义优化器的类型和参数。

```python
optimizer=dict(
    type=AdamW,  # 优化器类型
    lr=base_lr,  # 学习率
    weight_decay=0.05,  # 权重衰减
)
```

## 优化器封装

优化器封装模块用于进一步配置优化器的参数调整策略和其他高级功能。

```python
optim_wrapper=dict(
    type=OptimWrapper,  # 优化器封装类型
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.05),  # 优化器配置
    paramwise_cfg=dict(
        norm_decay_mult=0,  # 归一化层权重衰减倍数
        bias_decay_mult=0,  # 偏置项权重衰减倍数
        bypass_duplicate=True,  # 是否绕过重复参数
    ),
)
```

#### 优化器参数调整策略

优化器参数调整策略模块用于动态调整优化器的参数。

```python
param_scheduler=[
    dict(
        type=LinearLR,  # 线性学习率调整
        start_factor=1.0e-5,  # 起始因子
        by_epoch=False,  # 是否按 epoch 调整
        begin=0,  # 开始调整的迭代
        end=1000,  # 结束调整的迭代
    ),
    dict(
        type=CosineAnnealingLR,  # 余弦退火学习率调整
        eta_min=base_lr * 0.05,  # 最小学习率
        begin=max_epochs // 2,  # 开始调整的 epoch
        end=max_epochs,  # 结束调整的 epoch
        T_max=max_epochs // 2,  # 余弦周期
        by_epoch=True,  # 是否按 epoch 调整
        convert_to_iter_based=True,  # 是否转换为迭代调整
    ),
]
```
