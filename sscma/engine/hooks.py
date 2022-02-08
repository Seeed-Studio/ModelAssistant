"""This module is used to implement and register the custom Hooks.

The default implementation only does the register process. Users need to rename
the ``CustomHook`` to the real name of the hook and implement it.
"""

from mmengine.hooks import Hook

from sscma.registry import HOOKS


@HOOKS.register_module()
class CustomHook(Hook):
    """Subclass of `mmengine.Hook`.

    Warning:
        The class attribute ``priority`` will influence the excutation sequence
        of other hooks.
    """
    priority = 'NORMAL'
    ...
