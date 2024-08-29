from mmengine.config import read_base

with read_base():
    from .datasets.lancedb_bs32 import *
    from ._base_.default_runtime import *
    from .models.timm_classify import *
    from .schedules.AdamW_linear_coslr_bs2048 import *

train_dataloader.batch_size=64
auto_scale_lr = dict(base_batch_size=64)
model.model_name="mobilenetv4_hybrid_medium"