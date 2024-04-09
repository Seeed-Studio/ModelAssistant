# Copyright (c) Seeed Tech Ltd. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.structures import DetDataSample, OptSampleList
from mmengine.optim import OptimWrapper

from sscma.registry import MODELS

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class Fomo(SingleStageDetector):
    def __init__(
        self,
        backbone: Dict,
        neck: Optional[Dict] = None,
        head: Optional[Dict] = None,
        data_preprocessor: Optional[Dict] = None,
        skip_preprocessor: bool = False,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        pretrained: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ):
        # data_preprocessor=None
        self.skip_preprocessor = skip_preprocessor
        super().__init__(backbone, neck, head, train_cfg, test_cfg, data_preprocessor, init_cfg)

    def forward(self, inputs: torch.Tensor, data_samples: OptSampleList = None, mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

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

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0).to(self.data_preprocessor.device)
        if self.skip_preprocessor:
            if inputs.dtype == torch.uint8:
                inputs = inputs / 255
            if inputs.dtype == torch.int8:
                inputs = inputs / 128

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' 'Only supports loss, predict and tensor mode')

    def _forward(self, batch_inputs, batch_data_samples):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return torch.softmax(results[0].permute(0, 2, 3, 1), dim=-1)

    def train_step(self, data: Union[dict, tuple, list], optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            if not self.skip_preprocessor:
                data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: [tuple, dict, list]) -> list:
        if not self.skip_preprocessor:
            data = self.data_preprocessor(data, False)
        return self._run_forward(data, 'predict')

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        if not self.skip_preprocessor:
            data = self.data_preprocessor(data, False)
        return self._run_forward(data, 'predict')
