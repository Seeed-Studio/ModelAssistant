from typing import Any, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric


class MseMetric(BaseMetric):
    """The MSE metric for evaluating the quality of the predictions.

    Args:
        dist_sync_on_step (bool): Whether to synchronize the metric states
            across processes at each ``forward()`` call. Defaults to False.
    """

    def __init__(self):
        super().__init__()

    def add(self, pred: float, target: float) -> None:
        """Add the metric result.

        Args:
            pred (float): The predicted value.
            target (float): The ground truth value.
        """
        self._results.append((pred - target) ** 2)

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        self.results.append(data_samples[0]["loss"].cpu().numpy())

    def compute(self) -> float:
        """Compute the metric.

        Returns:
            float: The computed metric.
        """
        return dict(mse=np.sum(self.results) / len(self.results))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics.
        """
        return dict(mse=np.sum(self.results) / len(self.results))
