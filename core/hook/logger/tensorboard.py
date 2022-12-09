import os.path as osp
from typing import Optional, OrderedDict, Dict, Union

import torch
from mmcv.runner import HOOKS
from mmcv.fileio.file_client import FileClient
from mmcv.runner.dist_utils import master_only
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.hooks.logger.text import TextLoggerHook

from core.hook.logger.text import TextLoggerHook


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
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
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

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

        if self.out_dir is not None:
            self.file_client = FileClient.infer_client(self.file_client_args,
                                                       self.out_dir)
            # The final `self.out_dir` is the concatenation of `self.out_dir`
            # and the last level directory of `runner.work_dir`
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)
            runner.logger.info(
                f'Text logs will be saved to {self.out_dir} by '
                f'{self.file_client.name} after the training process.')

        self.start_iter = runner.iter
        self.json_log_path = osp.join(runner.work_dir,
                                      f'{runner.timestamp}.log.json')
        if runner.meta is not None:
            self._dump_log(runner.meta, runner)

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(mode=self.get_mode(runner),
                               epoch=self.get_epoch(runner),
                               iter=cur_iter)

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        # if 'time' in runner.log_buffer.output:
        # statistic memory
        if torch.cuda.is_available():
            log_dict['memory'] = self._get_max_memory(runner)
        log_dict = dict(log_dict, **runner.log_buffer.output)  # type: ignore
        self.log_dict = log_dict
        self._log_info(log_dict, runner)
        self._dump_log(log_dict, runner)
        return log_dict

    @master_only
    def after_run(self, runner) -> None:
        self.writer.close()
