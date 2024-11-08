import copy

import torch

from mmengine import MODELS
from mmengine.model import BaseModel
from sscma.utils.typing_utils import OptConfigType, OptMultiConfig
from ..backend import BaseInfer


class AnomalyInfer(BaseModel):
    """AnomalyInfer class for Anomaly serial inference."""

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
                f'Invalid mode "{mode}". ' "AnomalyetInfer Only supports predict mode"
            )

    def _predict(self, inputs: torch.Tensor, batch_data_samples=None):
        data = self.func.infer(inputs, split=False)
        res = []
        for d in data:
            res.append(torch.from_numpy(d))
        loss1, loss2, loss3 = self.vae_model.loss_function(*res)
        loss = loss1 + loss2 + loss3
        return [dict(loss=loss)]

    def set_infer(self, func: BaseInfer, Config: OptConfigType = None):
        self.func = func
        self.func.load_weights()
        if Config is not None:
            self.config = copy.deepcopy(Config)
            self.vae_model = MODELS.build(self.config.model)
