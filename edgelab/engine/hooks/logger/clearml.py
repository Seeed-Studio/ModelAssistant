from typing import Dict, Optional, Union

from mmengine.dist.utils import master_only

from edgelab.registry import HOOKS

from .text import TextLoggerHook

# from mmcv.runner import HOOKS
# from mmcv.runner.dist_utils import master_only


@HOOKS.register_module(force=True)
class ClearMLLoggerHook(TextLoggerHook):
    def __init__(
        self,
        by_epoch: bool = True,
        interval: int = 10,
        ignore_last: bool = True,
        reset_flag: bool = False,
        interval_exp_name: int = 1000,
        out_dir: Optional[str] = None,
        out_suffix: Union[str, tuple] = ...,
        keep_local: bool = True,
        ndigits: int = 4,
        init_kwargs: Optional[Dict] = None,
        file_client_args: Optional[Dict] = None,
    ):
        super().__init__(
            by_epoch,
            interval,
            ignore_last,
            reset_flag,
            interval_exp_name,
            out_dir,
            out_suffix,
            keep_local,
            ndigits,
            file_client_args,
        )
        try:
            import clearml
        except ImportError:
            raise ImportError('Please run "pip install clearml" to install clearml')
        self.clearml = clearml
        self.init_kwargs = init_kwargs

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        task_kwargs = self.init_kwargs if self.init_kwargs else {}
        self.task = self.clearml.Task.init(**task_kwargs)
        self.task_logger = self.task.get_logger()

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        for tag, val in tags.items():
            self.task_logger.report_scalar(tag, tag, val, self.get_iter(runner))

        return super().log(runner)
