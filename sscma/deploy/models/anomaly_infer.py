import torch

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
                f'Invalid mode "{mode}". ' "RTMDetInfer Only supports predict mode"
            )

    def _predict(self, inputs: torch.Tensor, batch_data_samples=None):
        data = self.func.infer(inputs)[0]
        mse = torch.mean((torch.from_numpy(data[0]) - inputs) ** 2)
        return [dict(loss=mse)]

    def set_infer(self, func: BaseInfer, Config: OptConfigType = None):
        self.func = func
        self.func.load_weights()
