from mmengine.config import read_base
with read_base():
    from .datasets.imagenet_bs32 import *
    from ._base_.default_runtime import *
    from .models.timm_resnet50 import *
    from .schedules.imagenet_bs256 import *

