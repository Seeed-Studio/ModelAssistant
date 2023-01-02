from .classfication_loss import LabelSmoothCrossEntropyLoss
from .pfld_loss import PFLDLoss
from .detecter_loss import DetectorLoss
from .nll_loss import NLLLoss
from .bce_withlogits_loss import BCEWithLogitsLoss

__all__ = [
    'LabelSmoothCrossEntropyLoss', 'PFLDLoss', 'DetectorLoss', 'NLLLoss',
    'BCEWithLogitsLoss'
]
