### 训练示例

对于训练可使用openmim项目下的mim工具，在以上执行`pip install openmim`命令时便已经安装了mim工具，此时可使用如下命令训练相应模型。


<details>
<summary>单机单 GPU：</summary>

```bash
# 训练分类模型
mim train mmdet $CONFIG --work-dir work_dir

# 测试分类模型
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)"

# 训练目标检测/实例分割模型
mim train mmdet $CONFIG --work-dir $WORK_DIR

# 测试目标检测/实例分割模型
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm

# 训练语义分割模型
mim train mmseg $CONFIG --work-dir $WORK_DIR

# 测试语义分割模型
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU
```

#### 参数解释

- `$CONFIG`: `configs/` 文件夹下的配置文件路径
- `$WORK_DIR`: 用于保存日志和模型权重文件的文件夹
- `$CHECKPOINT`: 权重文件路径

**提示：** 使用参数`--gpus=0`可转为CPU进行训练。

</details>
<details>
<summary>单机多 GPU （以 4 GPU 为例）：</summary>

```bash
# 训练分类模型
mim train mmcls $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4

# 测试分类模型
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)" --launcher pytorch --gpus 4

# 训练目标检测/实例分割模型
mim train mmdet $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4

# 测试目标检测/实例分割模型
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm --launcher pytorch --gpus 4

# 训练语义分割模型
mim train mmseg $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4 

# 测试语义分割模型
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU --launcher pytorch --gpus 4
```

#### 参数解释

- `$CONFIG`: `configs/` 文件夹下的配置文件路径
- `$WORK_DIR`: 用于保存日志和模型权重文件的文件夹
- `$CHECKPOINT`: 权重文件路径

</details>
<details>
<summary>多机多 GPU （以 2 节点共计 16 GPU 为例）</summary>

```bash
# 训练分类模型
mim train mmcls $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 测试分类模型
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)" --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 训练目标检测/实例分割模型
mim train mmdet $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 测试目标检测/实例分割模型
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 训练语义分割模型
mim train mmseg $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 测试语义分割模型
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

#### 参数解释

- `$CONFIG`: `configs/` 文件夹下的配置文件路径
- `$WORK_DIR`: 用于保存日志和模型权重文件的文件夹
- `$CHECKPOINT`: 模型权重文件路径
- `$PARTITION`: 使用的 Slurm 分区

</details>