import mmengine
import mmengine.runner.runner as runner
from mmengine.utils import digit_version

from .evaluation import Evaluator
from .version import __version__, version_info

runner.Evaluator = Evaluator

mmengine_minimum_version = '0.3.0'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

__all__ = ['__version__', 'version_info', 'digit_version']
