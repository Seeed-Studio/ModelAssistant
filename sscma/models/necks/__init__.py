# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from .fpn import FPN
from .gap import GlobalAveragePooling
from .spp import SPP

__all__ = ['SPP', 'FPN', 'GlobalAveragePooling']
