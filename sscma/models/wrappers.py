"""This module is used to implement and register the custom model wrapper.

Since ``mmengine.runner.Runner`` will call ``train_step``, ``val_step`` and
``test_step`` in different phases. There should be a wrapper on
``DistributedDataParallel``, ``FullyShardedDataParallel`` .etc. to implements
these methods. MMEngine has provided the commonly used wrappers for users, but
you can still customize the wrapper for some special requirements.

The default implementation only does the register process. Users need to rename
the ``CustomWrapper`` to the real name of the wrapper and implement it.
"""


class CustomWrapper:
    ...
