# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, no_type_check

from .dist_backends import BaseDistBackend, NonDist, TorchCPUDist, TorchCUDADist

_DIST_BACKENDS = {
    "non_dist": NonDist,
    "torch_cpu": TorchCPUDist,
    "torch_cuda": TorchCUDADist,
}

_DEFAULT_BACKEND = "non_dist"

# Caching created dist backend instances
_DIST_BACKEND_INSTANCES: dict = {}


def list_all_backends() -> List[str]:
    """Returns a list of all distributed backend names.

    Returns:
        List[str]: A list of all distributed backend names.
    """
    return list(_DIST_BACKENDS.keys())


def set_default_dist_backend(dist_backend: str) -> None:
    """Set the given distributed backend as the default distributed backend.

    Args:
        dist_backend (str): The distribute backend name to set.
    """
    assert dist_backend in _DIST_BACKENDS
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = dist_backend


@no_type_check
def get_dist_backend(dist_backend: Optional[str] = None) -> BaseDistBackend:
    """Returns distributed backend by the given distributed backend name.

    Args:
        dist_backend (str, optional): The distributed backend name want to get.
            if None, return the default distributed backend.

    Returns:
        :obj:`BaseDistBackend`: The distributed backend instance.
    """
    if dist_backend is None:
        dist_backend = _DEFAULT_BACKEND
    assert dist_backend in _DIST_BACKENDS
    if dist_backend not in _DIST_BACKEND_INSTANCES:
        dist_backend_cls = _DIST_BACKENDS[dist_backend]
        _DIST_BACKEND_INSTANCES[dist_backend] = dist_backend_cls()
    return _DIST_BACKEND_INSTANCES[dist_backend]
