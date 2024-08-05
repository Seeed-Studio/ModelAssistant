from mmengine.config import read_base

with read_base():
    from .datasets.imagenet_bs32 import *
    from ._base_.default_runtime import *
    from .models.timm_classify import *
    from .schedules.sgd_linear_coslr_bs2048 import *

model.model_name = "resnet50"
