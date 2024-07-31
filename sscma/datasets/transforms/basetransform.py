# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import inspect
import functools
import copy
import weakref



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
    if getattr(cls, '_methods_with_randomness', None):
        raise RuntimeError(
            f'Class {cls.__name__} decorated with '
            '``avoid_cache_randomness`` should not have methods decorated '
            'with ``cache_randomness`` (invalid methods: '
            f'{cls._methods_with_randomness})')

    class AvoidCacheRandomness:

        def __get__(self, obj, objtype=None):
            # Here we check the value in `objtype.__dict__` instead of
            # directly checking the attribute
            # `objtype._avoid_cache_randomness`. So if the base class is
            # decorated with :func:`avoid_cache_randomness`, it will not be
            # inherited by subclasses.
            return objtype.__dict__.get('_avoid_cache_randomness', False)

    cls.avoid_cache_randomness = AvoidCacheRandomness()
    cls._avoid_cache_randomness = True

    return cls



class cache_randomness:
    """Decorator that marks the method with random return value(s) in a
    transform class.

    This decorator is usually used together with the context-manager
    :func`:cache_random_params`. In this context, a decorated method will
    cache its return value(s) at the first time of being invoked, and always
    return the cached values when being invoked again.

    .. note::
        Only an instance method can be decorated with ``cache_randomness``.
    """

    def __init__(self, func):

        # Check `func` is to be bound as an instance method
        if not inspect.isfunction(func):
            raise TypeError('Unsupport callable to decorate with'
                            '@cache_randomness.')
        func_args = inspect.getfullargspec(func).args
        if len(func_args) == 0 or func_args[0] != 'self':
            raise TypeError(
                '@cache_randomness should only be used to decorate '
                'instance methods (the first argument is ``self``).')

        functools.update_wrapper(self, func)
        self.func = func
        self.instance_ref = None

    def __set_name__(self, owner, name):
        # Maintain a record of decorated methods in the class
        if not hasattr(owner, '_methods_with_randomness'):
            setattr(owner, '_methods_with_randomness', [])

        # Here `name` equals to `self.__name__`, i.e., the name of the
        # decorated function, due to the invocation of `update_wrapper` in
        # `self.__init__()`
        owner._methods_with_randomness.append(name)

    def __call__(self, *args, **kwargs):
        # Get the transform instance whose method is decorated
        # by cache_randomness
        instance = self.instance_ref()
        name = self.__name__

        # Check the flag ``self._cache_enabled``, which should be
        # set by the contextmanagers like ``cache_random_parameters```
        cache_enabled = getattr(instance, '_cache_enabled', False)

        if cache_enabled:
            # Initialize the cache of the transform instances. The flag
            # ``cache_enabled``` is set by contextmanagers like
            # ``cache_random_params```.
            if not hasattr(instance, '_cache'):
                setattr(instance, '_cache', {})

            if name not in instance._cache:
                instance._cache[name] = self.func(instance, *args, **kwargs)
            # Return the cached value
            return instance._cache[name]
        else:
            # Clear cache
            if hasattr(instance, '_cache'):
                del instance._cache
            # Return function output
            return self.func(instance, *args, **kwargs)

    def __get__(self, obj, cls):
        self.instance_ref = weakref.ref(obj)
        # Return a copy to avoid multiple transform instances sharing
        # one `cache_randomness` instance, which may cause data races
        # in multithreading cases.
        return copy.copy(self)
    

class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self,
                 results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @abstractmethod
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
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
