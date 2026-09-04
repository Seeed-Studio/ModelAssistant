# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
"""Inference backends.

Backends are imported lazily (PEP 562) so that importing this package does
not pull heavyweight, backend-specific dependencies (e.g. ``tensorflow`` for
TFLite/SavedModel or ``onnxruntime`` for ONNX) unless the backend is
actually used.
"""

import importlib
from typing import TYPE_CHECKING

_LAZY_MODULES = {
    'BaseInfer': '.base_infer',
    'OnnxInfer': '.onnxruntime_infer',
    'TorchScriptInfer': '.torchscript_infer',
    'SavedModelInfer': '.saved_model_infer',
    'TFliteInfer': '.tflite_infer',
    'HailoInfer': '.hailo_infer',
}

__all__ = list(_LAZY_MODULES.keys())


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = getattr(module, name)
        return globals()[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from .base_infer import BaseInfer
    from .hailo_infer import HailoInfer
    from .onnxruntime_infer import OnnxInfer
    from .saved_model_infer import SavedModelInfer
    from .tflite_infer import TFliteInfer
    from .torchscript_infer import TorchScriptInfer
