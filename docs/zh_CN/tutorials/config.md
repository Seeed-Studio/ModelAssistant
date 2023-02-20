#  模型配置

- [模型配置](#模型配置)
    - [配置文件路径结构](#配置文件路径结构)
    - [配置文件内容](#配置文件内容)
        - [重要参数](#重要参数)
        - [网络模型配置](#网络模型配置)
        - [数据集和验证配置](#数据集和验证配置)
        - [优化器配置](#优化器配置)
        - [配置文件继承](#配置文件继承)
    - [通过脚本参数修改配置文件](#通过脚本参数修改配置文件)
    - [FAQs](#faqs)

Edgelab存储库使用OpenMMLab提供的配置系统，模块化、继承性设计，便于进行各种实验。您可以在[这里](../../../configs/)检查配置文件。


## 配置文件路径结构

存储库中配置文件的目录结构如下：
```sh
configs  
├── audio_classify  
│   ├── ali_classiyf_small_8k_8192.py  
│   └── README.md  
├── _base_  
│   ├── datasets  
│   │   └── coco_detection.py  
│   ├── default_runtime.py  
│   └── schedules  
│       ├── schedule_1x.py  
│       ├── schedule_20e.py  
│       └── schedule_2x.py  
├── fastestdet  
│   ├── fastestdet_shuffv2_spp_fomo_voc.py  
│   └── fastestdet_shuffv2_spp_voc.py  
├── fomo  
│   ├── fomo_mobnetv2_0.35_x8_abl_coco.py  
│   ├── fomo_mobnetv2_x8_coco.py  
│   └── fomo_mobnetv2_x8_voc.py  
├── pfld  
│   ├── pfld_mv2n_112.py  
│   └── README.md  
└── yolo  
    ├── README.md  
    └── yolov3_mbv2_416_voc.py  
```

## 配置文件内容

以[fomo_mobnetv2_0.35_x8_abl_coco.py](../../../configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py)为例，我们将根据不同的功能模块介绍config中的各个字段：

### 重要参数

更改训练配置时，通常需要修改以下参数。例如，`height`和`width`参数通常用于图像大小。所以我们建议在配置文件中单独定义这些参数。

```sh
height=96       # 输入图像高度
width=96        # 输入图像宽度
batch_size=16   # 验证期间单个 GPU 的批量大小
workers=4       # 验证期间单个GPU预读取数据的线程数
epoches=300     # 最大训练轮次：300轮
lr=0.001        # 学习率
```

### 网络模型配置

在我们的存储库中，我们使用模型来设置检测算法组件，包括backbone、neck等神经网络组件。部分模型配置如下：
```sh
num_classes=2   # 类别数
model = dict(
    type='Fomo',    # 检测器名称
    backbone=dict(type='MobileNetV2', widen_factor=0.35, out_indices=(2, )),    # 主干网络配置
    head=dict(
        type='Fomo_Head',   # 检测头配置
        input_channels=16,  # 输入通道数，与neck的输入通道一致
        num_classes=num_classes,    # 分类的类别数
        middle_channels=[96, 32],   # 检测头卷积的输出通道数
        act_cfg='ReLU6',    # 激活函数配置
        loss_cls=dict(type='BCEWithLogitsLoss', # Sigmoid层和BCELoss结合在一起的损失函数
                      reduction='none',
                      pos_weight=40),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
        cls_weight=40,  # pos_weight的参数值 
    ),
)
```

### 数据集和验证配置

需要设置数据集和数据管道来构建数据加载器。由于这部分的复杂性，我们使用中间变量来简化数据加载器配置的编写。更复杂的数据增强方法可以在[pipiline目录](../../../edgelab/datasets/pipelines/)中找到。我们将在这里演示fomo的训练和测试数据流。
```sh
dataset_type = 'FomoDatasets'   # 数据集类型，用于定义数据集
data_root = ''  # 数据根目录
train_pipeline = [  # 训练数据加载管道
    dict(type='RandomResizedCrop', height=height, width=width, scale=(0.90, 1.1),
         p=1),  # RandomResizedCrop数据增强
    dict(type='Rotate', limit=20), # 最大度数为20的旋转变换
    dict(type='RandomBrightnessContrast',   # RandomBrightnessContrast数据增强
         brightness_limit=0.2,  # 亮度改变的系数范围
         contrast_limit=0.2,    # 对比度改变的系数范围
         p=0.5),    # 使用RandomBrightnessContrast数据增强的概率
    dict(type='HorizontalFlip', p=0.5), # 围绕y轴水平翻转
]
test_pipeline = [dict(type='Resize', height=height, width=width, p=1)] # 将输入调整为给定的高度和宽度

data = dict(    # 训练数据加载器配置
            samples_per_gpu=batch_size, # 训练期间单个 GPU 的批量大小
            workers_per_gpu=workers,    # 验证期间单个GPU预读取数据的线程数
            train_dataloader=dict(collate=True),   # 是否生成小批次样本
            val_dataloader=dict(collate=True),
            train=dict(type=dataset_type,
                       data_root=data_root,
                       ann_file='annotations/person_keypoints_train2017.json',  # 标注文件路径
                       img_prefix='train2017',  # 图片路径
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     data_root=data_root,
                     test_mode=True, # 打开数据集的测试模式以避免过滤标注或图像。
                     ann_file='annotations/person_keypoints_val2017.json',
                     img_prefix='val2017',
                     pipeline=test_pipeline),
            test=dict(type=dataset_type,
                      data_root=data_root,
                      test_mode=True,
                      ann_file='annotations/person_keypoints_val2017.json',
                      img_prefix='val2017',
                      pipeline=test_pipeline))
```

评估器用于计算训练模型在验证和测试数据集上的指标。评估者的配置由一个或一系列指标配置组成：
```sh
evaluation = dict(interval=1, metric=['mAP'], fomo=True)    # 验证评估指标，mAP评估
find_unused_parameters = True
```

### 优化器配置

```sh
optimizer = dict(type='Adam', lr=lr, weight_decay=0.0005) # 具有基础学习率和权重衰减的Adam梯度下降优化器

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))   # 用于构建优化器hook的配置，更多应用细节参考https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8
```

### 配置文件继承

`config/_base_/default_runtime.py`包含默认运行配置。由_base_ 中的组件组成的配置称为原语。  
为了便于理解，我们建议使用者继承现有的方法。例如，在fomo配置文件中设置`__base__='../_base_/default_runtime.py'`，然后修改配置文件中的必要字段。
```sh
checkpoint_config = dict(interval=5)    # 配置设置检查点hook，更多细节请参考https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint，保存间隔为1
log_config = dict(  # 配置注册记录器hook
    interval=150,   # 打印日志间隔
    hooks=[
        # dict(type='TextLoggerHook', ndigits=4),   # txt文本日志
        dict(type='TensorboardLoggerHook', ndigits=4)   # tensorboard日志
    ])  # 记录训练过程的日志

epochs=300
runner = dict(type='EpochBasedRunner',  # 使用的runner类型(例如IterBasedRunner或者EpochBasedRunner)
              max_epochs=epochs)    # runner运行max_epochs次的工作流，对于IterBasedRunner使用`max_iters`

dist_params = dict(backend='nccl')  # 设置分布式训练的参数，也可以设置端口

log_level = 'INFO'  # 日志等级

load_from = None    # 从给定路径加载模型作为预训练模型，不会恢复训练
resume_from = None  # 从给定路径恢复检查点，训练将从保存检查点的轮次恢复

workflow = [('train', 1)]   # runner的工作流程。[('train', 1)]表示只有一个工作流，名为'train'的工作流执行一次。工作流根据 total_epochs训练模型300轮

opencv_num_threads = 1  # # 启动opencv多线程

# 将多进程启动方法设置为`fork`以加速训练
# mp_start_method = 'fork'

work_dir = './work_dirs'    # 用于保存当前实验的模型检查点和日志的目录
```

## 通过脚本参数修改配置文件

使用tools/train.py或tools/test.py提交作业时，可以指定--cfg-options就地修改配置。

- 更新字典链的配置键  
可以按照原始配置中字典键的顺序指定配置选项。例如，`--cfg-options data_root='./dataset/coco'`更改数据集的数据根目录。


## FAQs
不同模型的配置文件会有一定的差异, 更多细节请参考[mmdet config](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html)，[mmpose config](https://mmpose.readthedocs.io/en/latest/tutorials/0_config.html)和[mmcls config](https://mmclassification.readthedocs.io/en/latest/tutorials/config.html)。