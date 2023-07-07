from torch.utils.data import Dataset

from edgelab.registry import DATASETS


@DATASETS.register_module()
class FoMoDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def get_preceicn_recall_f1(self):
        """Calculate the predicted evaluation index through the output of the model"""
        pass

    def eval(self):
        pass
