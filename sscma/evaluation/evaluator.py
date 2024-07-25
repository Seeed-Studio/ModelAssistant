"""This module is used to patch the default Evaluator in MMEngine.

For the convenient of customizing the Metric, ``sscma`` patch
the ``mmengine.evaluator.Evaluator`` with the local ``Evaluator`` Thanks to
this, the Metric in sscma only needs to implement the ``add`` and
``compute_metric``.

Warning:
    If there is a need to customize the Evaluator for more complicated evaluate
    process. The methods defined in ``CustomEvaluator`` must call
    ``metric.compute`` and ``metric.add`` rather than ``metric.process`` and
    ``metric.evaluate``
"""

from mmengine.evaluator import Evaluator as MMEngineEvaluator
from mmengine.structures import BaseDataElement



class Evaluator(MMEngineEvaluator):

    def process(self, data_samples, data_batch=None):
        _data_samples = []
        for data_sample in data_samples:
            if isinstance(data_sample, BaseDataElement):
                _data_samples.append(data_sample.to_dict())
            else:
                _data_samples.append(data_sample)

        for metric in self.metrics:
            metric.add(data_batch, _data_samples)

    def evaluate(self, size):
        metrics = {}
        for metric in self.metrics:
            _results = metric.compute(size)

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)
        return metrics
