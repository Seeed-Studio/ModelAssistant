# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union


def avoid_cache_randomness(cls):
    """Decorator that marks a data transform class (subclass of
    :class:`BaseTransform`) prohibited from caching randomness. With this
    decorator, errors will be raised in following cases:

        1. A method is defined in the class with the decorate
    `cache_randomness`;
        2. An instance of the class is invoked with the context
    `cache_random_params`.

    A typical usage of `avoid_cache_randomness` is to decorate the data
    transforms with non-cacheable random behaviors (e.g., the random behavior
    can not be defined in a method, thus can not be decorated with
    `cache_randomness`). This is for preventing unintentinoal use of such data
    transforms within the context of caching randomness, which may lead to
    unexpected results.
    """

    # Check that cls is a data transform class
    assert issubclass(cls, BaseTransform)

    # Check that no method is decorated with `cache_randomness` in cls
    if getattr(cls, "_methods_with_randomness", None):
        raise RuntimeError(
            f"Class {cls.__name__} decorated with "
            "``avoid_cache_randomness`` should not have methods decorated "
            "with ``cache_randomness`` (invalid methods: "
            f"{cls._methods_with_randomness})"
        )

    class AvoidCacheRandomness:
        def __get__(self, obj, objtype=None):
            # Here we check the value in `objtype.__dict__` instead of
            # directly checking the attribute
            # `objtype._avoid_cache_randomness`. So if the base class is
            # decorated with :func:`avoid_cache_randomness`, it will not be
            # inherited by subclasses.
            return objtype.__dict__.get("_avoid_cache_randomness", False)

    cls.avoid_cache_randomness = AvoidCacheRandomness()
    cls._avoid_cache_randomness = True

    return cls


class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        return self.transform(results)

    @abstractmethod
    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """The transform function. All subclass of BaseTransform should
        override this method.
        This function takes the result dict as the input, and can add new
        items to the dict or modify existing items in the dict. And the result
        dict will be returned in the end, which allows to concate multiple
        transforms into a pipeline.
        Args:
            results (dict): The result dict.
        Returns:
            dict: The result dict.
        """
