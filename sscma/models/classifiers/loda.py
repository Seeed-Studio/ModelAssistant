# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import math
from typing import List, Optional, Union

import torch
from mmengine.model import BaseModel

from sscma.registry import MODELS
from sscma.structures import ClsDataSample


@MODELS.register_module()
class LODA(BaseModel):
    def __init__(
        self,
        num_bins: int = 10,
        num_cuts: int = 100,
        yield_rate=0.9,
        init_cfg: Union[dict, List[dict], None] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.num_bins = num_bins
        self.num_cuts = num_cuts
        self.weights = torch.ones(num_cuts, dtype=torch.float) / num_cuts

        self.histograms = torch.zeros((self.num_cuts, self.num_bins))
        self.limits = torch.zeros((self.num_cuts, self.num_bins + 1))
        self.yield_rate = yield_rate
        self.num = 0

    def parameters(self, recurse: bool = True):
        """Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
        """
        yield torch.nn.Parameter(torch.randn(1))

    def forward(
        self, inputs: torch.Tensor, data_samples: Optional[List[ClsDataSample]] = None, mode: str = 'tensor'
    ) -> Union[dict, List]:
        if mode == 'tensor':
            return self.loss(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise ValueError(f'Invalid mode "{mode}".')

    def loss(self, inputs: torch.Tensor, data_samples: Optional[List[ClsDataSample]] = None) -> dict:
        x = inputs[0] if isinstance(inputs, list) else inputs
        x = x.to('cpu')
        pred_scores = torch.zeros(x.shape[0], 1)

        num_features = x.shape[1]

        num_features_sqrt = int(math.sqrt(num_features))
        num_features_zero = num_features - num_features_sqrt

        self.projections_ = torch.randn(self.num_cuts, num_features)

        for i in range(self.num_cuts):
            perm = torch.randperm(num_features)[:num_features_zero]
            self.projections_[i, perm] = 0.0
            projected_vectors = self.projections_[i, :].unsqueeze(0).matmul(x.T).squeeze(0)
            self.histograms[i, :], self.limits[i, :] = torch.histogram(
                projected_vectors, bins=self.num_bins, density=False
            )
            self.histograms[i, :] += 1e-12
            self.histograms[i, :] /= torch.sum(self.histograms[i, :])

            inds = torch.searchsorted(self.limits[i, : self.num_bins - 1], projected_vectors, side='left')
            pred_scores[:, 0] += -self.weights[i] * torch.log(self.histograms[i, inds])

        decision_scores: torch.Tensor = (pred_scores / self.num_cuts).ravel() * 1.06
        self.threshold_: torch.Tensor = torch.quantile(decision_scores, self.yield_rate)
        return {
            'loss': torch.nn.Parameter(self.threshold_),
            'threshold': self.threshold_,
        }

    def predict(self, inputs: torch.Tensor, data_samples: Optional[List[ClsDataSample]] = None) -> List[ClsDataSample]:
        x = inputs[0] if isinstance(inputs, list) else inputs
        x = x.to('cpu')
        pred_scores = torch.zeros(x.shape[0], 1)
        for i in range(self.num_cuts):
            projected_vectors = self.projections_[i, :].unsqueeze(0).matmul(x.T).squeeze(0)
            inds = torch.searchsorted(self.limits[i, : self.num_bins - 1], projected_vectors, side='left')
            pred_scores[:, 0] += -self.weights[i] * torch.log(self.histograms[i, inds])
        pred_scores = (pred_scores / self.num_cuts).ravel()
        prediction = (pred_scores > self.threshold_).long().ravel()
        data_samples[0].set_pred_label(pred_scores).set_pred_score(prediction)
        self.num += sum(prediction).item()

        return [data_samples[0]]
