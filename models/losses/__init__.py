from .classfication_loss import LabelSmoothCrossEntropyLoss
from .pfld_loss import PFLDLoss
from .nll_loss import NLLLoss
from .bce_withlogits_loss import BCEWithLogitsLoss
from .fomo_loss import FomoLoss

__all__ = [
    'LabelSmoothCrossEntropyLoss', 'PFLDLoss', 'NLLLoss',
    'BCEWithLogitsLoss', 'FomoLoss'
]
