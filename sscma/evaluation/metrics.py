"""This module is used to patch the default Evaluator in MMEngine.

Follow the `Guide <https://mmeval.readthedocs.io/en/latest/tutorials/custom_metric.html>_`
in MMEval to customize a Metric.

The default implementation only does the register process. Users need to rename
the ``CustomMetric`` to the real name of the metric and implement it.
"""  # noqa: E501

from .base_metric import BaseMetric

from sscma.registry import METRICS


@METRICS.register_module()
class CustomMetric(BaseMetric):

    def add(self, gt, preds):
        ...

    # NOTE for evaluator
    def compute_metric(self, size):
        ...
