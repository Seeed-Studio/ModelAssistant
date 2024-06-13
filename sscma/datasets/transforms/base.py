import copy
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Union

from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import autocast_box_type
from mmengine.dataset import BaseDataset
from mmengine.dataset.base_dataset import Compose
from numpy import random


class BaseMixImageTransform(BaseTransform, metaclass=ABCMeta):

    def __init__(
        self,
        pre_transform: Optional[Sequence[str]] = None,
        prob: float = 1.0,
        use_cached: bool = False,
        max_cached_images: int = 40,
        random_pop: bool = True,
        max_refetch: int = 15,
    ):

        self.max_refetch = max_refetch
        self.prob = prob

        self.use_cached = use_cached
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.results_cache = []

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    @abstractmethod
    def get_indexes(self, dataset: Union[BaseDataset, list]) -> Union[list, int]:
        pass

    @abstractmethod
    def mix_img_transform(self, results: dict) -> dict:
        pass

    @autocast_box_type()
    def transform(self, results: dict) -> dict:

        if random.uniform(0, 1) > self.prob:
            return results

        assert 'dataset' in results
        dataset = results.pop('dataset', None)

        if self.use_cached:
            self.results_cache.append(copy.deepcopy(results))
            if len(self.results_cache) > self.max_cached_images:
                if self.random_pop:
                    index = random.randint(0, len(self.results_cache) - 1)
                else:
                    index = 0
                self.results_cache.pop(index)

        for _ in range(self.max_refetch):
            # get index of one or three other images
            if self.use_cached:
                indexes = self.get_indexes(self.results_cache)
                mix_results = [copy.deepcopy(self.results_cache[i]) for i in indexes]

            else:
                indexes = self.get_indexes(dataset)
                mix_results = [copy.deepcopy(dataset.get_data_info(i)) for i in indexes]

                if self.pre_transform is not None:
                    for i, data in enumerate(mix_results):
                        # pre_transform may also require dataset
                        data.update({'dataset': dataset})
                        # before Mosaic or MixUp need to go through
                        # the necessary pre_transform
                        _results = self.pre_transform(data)
                        _results.pop('dataset')
                        mix_results[i] = _results

            if None not in mix_results:
                results['mix_results'] = mix_results
                break

        else:
            raise RuntimeError(
                'The loading pipeline of the original dataset'
                ' always return None. Please check the correctness '
                'of the dataset and its pipeline.'
            )

        # Mosaic or MixUp
        results = self.mix_img_transform(results)

        if 'mix_results' in results:
            results.pop('mix_results')
        results['dataset'] = dataset

        return results
