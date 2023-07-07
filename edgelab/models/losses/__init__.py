from .classfication_loss import LabelSmoothCrossEntropyLoss
from .pfld_loss import PFLDLoss
from .nll_loss import NLLLoss
from .bce_withlogits_loss import BCEWithLogitsLoss

__all__ = ['LabelSmoothCrossEntropyLoss', 'PFLDLoss', 'NLLLoss', 'BCEWithLogitsLoss']
