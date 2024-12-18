import copy
import numpy as np
import torch

from mmengine import MODELS
from mmengine.model import BaseModel
from sscma.utils.typing_utils import OptConfigType, OptMultiConfig
from sscma.utils.misc import samplelist_boxtype2tensor
from ..backend import BaseInfer


class FomoInfer(BaseModel):
    """FomoInfer class for fomo serial inference."""

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.pred_head = None
        self.func = None
        self.config = None

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples=None,
        mode: str = "predict",
    ):
        """The unified entry for a forward process in both training and test.
        The method should accept three modes: "tensor", "predict" and "loss":

        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - ``mode="predict"``, return a list of :obj:`DetDataSample`.
        """
        if mode == "predict":
            return self._predict(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". ' "FomoInfer Only supports predict mode"
            )

    def _predict(self, inputs: torch.Tensor, batch_data_samples=None):
        data = self.func.infer(inputs)
        data = [torch.from_numpy(np.concatenate([d[0] for d in data], axis=0))]

        resutlts_dict = self.pred_head.predict_by_feat(data, batch_data_samples)

        for data_sample, pred_instances in zip(batch_data_samples, resutlts_dict):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(batch_data_samples)

        return batch_data_samples

    def set_infer(self, func: BaseInfer, Config: OptConfigType = None):
        self.func = func
        self.func.load_weights()
        if Config is not None:
            self.config = copy.deepcopy(Config)
            self.pred_head = MODELS.build(self.config.model.head)
