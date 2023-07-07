import os.path as osp
from typing import Optional, Union, Dict

from mmengine.utils import scandir
from edgelab.registry import HOOKS
from mmengine.dist.utils import master_only

# from mmcv.utils import scandir
# from mmcv.runner import HOOKS
# from mmcv.runner.dist_utils import master_only
from .text import TextLoggerHook


@HOOKS.register_module(force=True)
class WandbLoggerHook(TextLoggerHook):
    def __init__(
        self,
        init_kwargs: Optional[Dict] = None,
        commit: bool = True,
        by_epoch: bool = True,
        with_step: bool = True,
        log_artifact: bool = True,
        interval: int = 10,
        ignore_last: bool = True,
        reset_flag: bool = False,
        interval_exp_name: int = 1000,
        out_dir: Optional[str] = None,
        out_suffix: Union[str, tuple] = ...,
        keep_local: bool = True,
        ndigits: int = 4,
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
            import wandb
        except ImportError:
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb
        self.init_kwargs = init_kwargs
        self.commit = commit
        self.with_step = with_step
        self.log_artifact = log_artifact
        self.out_suffix = out_suffix

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)  # type: ignore
        else:
            self.wandb.init()  # type: ignore

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner)
        if tags:
            if self.with_step:
                self.wandb.log(tags, step=self.get_iter(runner), commit=self.commit)
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit)
        return super().log(runner)

    @master_only
    def after_run(self, runner) -> None:
        super().after_run(runner)
        if self.log_artifact:
            wandb_artifact = self.wandb.Artifact(name='artifacts', type='model')
            for filename in scandir(runner.work_dir, self.out_suffix, True):
                local_filepath = osp.join(runner.work_dir, filename)
                wandb_artifact.add_file(local_filepath)
            self.wandb.log_artifact(wandb_artifact)
        self.wandb.join()
