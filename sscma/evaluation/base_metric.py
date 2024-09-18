# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import Any, Dict, List, Optional

from .dist import get_dist_backend


class BaseMetric(metaclass=ABCMeta):
    """Base class for metric.

    To implement a metric, you should implement a subclass of ``BaseMetric``
    that overrides the ``add`` and ``compute_metric`` methods. ``BaseMetric``
    will automatically complete the distributed synchronization between
    processes.

    In the evaluation process, each metric will update ``self._results`` to
    store intermediate results after each call of ``add``. When computing the
    final metric result, the ``self._results`` will be synchronized between
    processes.

    Args:
        dataset_meta (dict, optional): Meta information of the dataset, this is
            required for some metrics that require dataset information.
            Defaults to None.
        dist_collect_mode (str, optional): The method of concatenating the
            collected synchronization results. This depends on how the
            distributed data is split. Currently only 'unzip' and 'cat' are
            supported. For PyTorch's ``DistributedSampler``, 'unzip' should
            be used. Defaults to 'unzip'.
        dist_backend (str, optional): The name of the distributed communication
            backend, you can get all the backend names through
            ``mmeval.core.list_all_backends()``.
            If ``None``, use the default backend. Defaults to None.
        logger (Logger, optional): The logger used to log messages.
            If ``None``, use the default logger of mmeval. Defaults to None.

    Example to implement an accuracy metric:

        >>> import numpy as np
        >>> from mmeval.core import BaseMetric
        >>>
        >>> class Accuracy(BaseMetric):
        ...     def add(self, predictions, labels):
        ...         self._results.append((predictions, labels))
        ...     def compute_metric(self, results):
        ...         predictions = np.concatenate([res[0] for res in results])
        ...         labels = np.concatenate([res[1] for res in results])
        ...         correct = (predictions == labels)
        ...         accuracy = sum(correct) / len(predictions)
        ...         return {'accuracy': accuracy}

    Stateless call of metric:

        >>> accuracy = Accuracy()
        >>> accuracy(predictions=[1, 2, 3, 4], labels=[1, 2, 3, 1])
        {'accuracy': 0.75}

    Accumulate batch:

        >>> for i in range(10):
        >>>     predicts = np.random.randint(0, 4, size=(10,))
        >>>     labels = predicts = np.random.randint(0, 4, size=(10,))
        >>>     accuracy.add(predicts, labels)
        >>> accuracy.compute()  # doctest: +SKIP
    """

    def __init__(
        self,
        dataset_meta: Optional[Dict] = None,
        dist_collect_mode: str = "unzip",
        dist_backend: Optional[str] = None,
        logger: Optional[Logger] = None,
    ):
        self.dataset_meta = dataset_meta
        assert dist_collect_mode in ("cat", "unzip")
        self.dist_collect_mode = dist_collect_mode
        self.dist_comm = get_dist_backend(dist_backend)
        self.logger = logger
        self._results: List[Any] = []

    @property
    def dataset_meta(self) -> Optional[Dict]:
        """Meta information of the dataset."""
        if self._dataset_meta is None:
            return self._dataset_meta
        else:
            return self._dataset_meta.copy()

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: Optional[Dict]) -> None:
        """Set the dataset meta information to the metric."""
        if dataset_meta is None:
            self._dataset_meta = dataset_meta
        else:
            self._dataset_meta = dataset_meta.copy()

    @property
    def name(self) -> str:
        """The metric name, defaults to the name of the class."""
        return self.__class__.__name__

    def reset(self) -> None:
        """Clear the metric stored results."""
        self._results.clear()

    def __call__(self, *args, **kwargs) -> Dict:
        """Stateless call for a metric compute."""
        cache_results = self._results
        self._results = []
        self.add(*args, **kwargs)
        metric_result = self.compute_metric(self._results)
        self._results = cache_results
        return metric_result

    def compute(self, size: Optional[int] = None) -> Dict:
        """Synchronize intermediate results and then call
        ``self.compute_metric``.

        Args:
            size (int, optional): The length of the entire dataset, it is only
                used when distributed evaluation. When batch size > 1, the
                dataloader may pad some data samples to make sure all ranks
                have the same length of dataset slice. The ``compute`` will
                drop the padded data based on this size.
                If None, do nothing. Defaults to None.

        Returns:
            dict: The computed metric results.
        """
        if not self.dist_comm.is_initialized or self.dist_comm.world_size == 1:
            return self.compute_metric(self._results)

        global_results = self.dist_comm.all_gather_object(self._results)

        collected_results: List[Any]
        if self.dist_collect_mode == "cat":
            # use `sum` to concatenate list
            # e.g. sum([[1, 3], [2, 4]], []) = [1, 3, 2, 4]
            collected_results = sum(global_results, [])
        else:
            collected_results = []
            for partial_result in zip(*global_results):
                collected_results.extend(list(partial_result))

        # NOTE: We use the given `size` to remove samples padded during
        # distributed evaluation. This requires that the size and order of
        # intermediate results stored in `self._results` should be consistent
        # with the evaluation samples.
        if size is not None:
            collected_results = collected_results[:size]

        if self.dist_comm.rank == 0:
            metric_result = self.compute_metric(collected_results)
        else:
            metric_result = None  # type: ignore

        global_metric_result = self.dist_comm.broadcast_object(metric_result, 0)
        return global_metric_result

    @abstractmethod
    def add(self, *args, **kwargs):
        """Override this method to add the intermediate results to
        ``self._results``.

        Note:
            For performance issues, what you add to the ``self._results``
            should be as simple as possible. But be aware that the intermediate
            results stored in ``self._results`` should correspond one-to-one
            with the samples, in that we need to remove the padded samples for
            the most accurate result.
        """

    @abstractmethod
    def compute_metric(self, results: List[Any]) -> Dict:
        """Override this method to compute the metric result from collectd
        intermediate results.

        The returned result of the metric compute should be a dictionary.
        """
