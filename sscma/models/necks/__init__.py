# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from .fpn import FPN
from .spp import SPP
from .gap import GlobalAveragePooling

__all__ = ['SPP', 'FPN', 'GlobalAveragePooling']
