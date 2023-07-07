import os
from typing import Optional, Dict, Union

from edgelab.registry import HOOKS
from mmengine.dist.utils import master_only
# from mmcv.runner import HOOKS

# from mmcv.runner.dist_utils import master_only
from mmengine.utils.dl_utils import TORCH_VERSION
# from mmcv.runner.hooks.logger.text import TextLoggerHook

from .text import TextLoggerHook


@HOOKS.register_module(force=True)
class TensorboardLoggerHook(TextLoggerHook):

    def __init__(self,
                 by_epoch: bool = True,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 interval_exp_name: int = 1000,
                 out_dir: Optional[str] = None,
                 out_suffix: Union[str, tuple] = ...,
                 keep_local: bool = True,
                 ndigits: int = 4,
                 file_client_args: Optional[Dict] = None):
        super().__init__(by_epoch, interval, ignore_last, reset_flag,
                         interval_exp_name, out_dir, out_suffix, keep_local,
                         ndigits, file_client_args)

        self.log_dir = out_dir

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)  # type: ignore
        if TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')
        self._tensorboard = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
        return super().log(runner)

    @master_only
    def after_run(self, runner) -> None:
        super().after_run(runner)
        self.writer.close()
