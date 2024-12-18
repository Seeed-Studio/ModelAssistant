# 模型配置

在训练深度学习任务时，我们通常需要定义一个模型来实现算法的主体。我们使用 `model` 字段用于配置模型的相关信息，包括模型的网络结构、尺寸大小、各部分的连接方式等，这是一种模块化的设计方式，使得用户可以根据自己的需求来定义模型，而各个模块的实现则定义在 `sscma` 的核心代码库中。


:::tip

对于模型核心，SSCMA 基于 MMEngine 开发，对模型的定义也遵循 MMEngine 的结构，其由执行器管理，且需要实现 train_step、val_step 和 test_step 方法。 对于检测、识别、分割一类的深度学习任务，上述方法通常为标准的流程，例如在 train_step 里更新参数，返回损失；val_step 和 test_step 返回预测结果。
:::

以 RTMDet 为例，我们分部分对模型配置进行说明。

## 数据预处理

定义输入数据的均值、标准差、颜色空间转换和数据增强策略。，我们可以在配置文件中定义数据预处理的参数，如均值、标准差、颜色空间转换等。

```python
data_preprocessor=dict(
    type=DetDataPreprocessor,  # 数据预处理器类型
    mean=[103.53, 116.28, 123.675],  # 均值
    std=[57.375, 57.12, 58.395],  # 标准差
    bgr_to_rgb=False,  # 是否将BGR转换为RGB
    batch_augments=[  # 批量数据增强
        dict(
            type=YOLOXBatchSyncRandomResize,  # 数据增强类型
            random_size_range=(224, 1024),  # 随机尺寸范围
            size_divisor=32,  # 尺寸除数
            interval=1,  # 迭代间隔
        )
    ],
)
```

## 主干网络

主干网络模块用于定义模型的主干结构。

```python
backbone=dict(
    type=TimmBackbone,  # 主干网络类型
    model_name="mobilenetv4_conv_small.e2400_r224_in1k",  # 模型名称
    features_only=True,  # 是否只提取特征
    pretrained=True,  # 是否使用预训练模型
    out_indices=[2, 3, 4],  # 输出特征层索引
    init_cfg=None,  # 初始化配置
)
```

## 颈部网络

颈部网络模块用于连接主干网络和头部网络。

```python
neck=dict(
    type=CSPNeXtPAFPN,  # 颈部网络类型
    deepen_factor=d_factor,  # 深度因子
    widen_factor=1,  # 宽度因子
    in_channels=[64, 96, 960],  # 输入通道数
    out_channels=256,  # 输出通道数
    num_csp_blocks=3,  # CSP块数量
    expand_ratio=0.5,  # 扩展比例
    norm_cfg=dict(type=SyncBatchNorm),  # 归一化配置
    act_cfg=dict(type=SiLU, inplace=True),  # 激活函数配置
)
```

## 头部网络

边界框头模块用于定义检测头的结构和损失函数。

```python
bbox_head=dict(
    type=RTMDetHead,  # 边界框头类型
    head_module=dict(
        type=RTMDetSepBNHeadModule,  # 头模块类型
        num_classes=80,  # 类别数量
        in_channels=256,  # 输入通道数
        stacked_convs=2,  # 堆叠卷积层数
        feat_channels=256,  # 特征通道数
        norm_cfg=dict(type=SyncBatchNorm),  # 归一化配置
        act_cfg=dict(type=SiLU, inplace=True),  # 激活函数配置
        share_conv=True,  # 是否共享卷积
        pred_kernel_size=1,  # 预测卷积核大小
        featmap_strides=[8, 16, 32],  # 特征图步长
    ),
    prior_generator=dict(type=MlvlPointGenerator, offset=0, strides=[8, 16, 32]),  # 先验框生成器
    bbox_coder=dict(type=DistancePointBBoxCoder),  # 边界框编码器
    loss_cls=dict(
        type=QualityFocalLoss, use_sigmoid=True, beta=2.0, loss_weight=1.0  # 分类损失
    ),
    loss_bbox=dict(type=GIoULoss, loss_weight=2.0),  # 边界框损失
)
```

## 训练配置

训练配置模块用于定义训练过程中的参数和策略。

```python
train_cfg=dict(
    assigner=dict(
        type=BatchDynamicSoftLabelAssigner,  # 分配器类型
        num_classes=num_classes,  # 类别数量
        topk=13,  # top-k 值
        iou_calculator=dict(type=BboxOverlaps2D),  # IOU 计算器
    ),
    allowed_border=-1,  # 允许的边界
    pos_weight=-1,  # 正样本权重
    debug=False,  # 是否开启调试
)
```

## 测试配置

测试配置模块用于定义测试过程中的参数和策略。

```python
test_cfg=dict(
    multi_label=True,  # 是否多标签
    nms_pre=30000,  # NMS 前的最大框数
    min_bbox_size=0,  # 最小边界框尺寸
    score_thr=0.001,  # 分数阈值
    nms=dict(type=nms, iou_threshold=0.65),  # NMS 配置
    max_per_img=300,  # 每张图片的最大检测框数
)
```


