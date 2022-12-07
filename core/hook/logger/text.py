import logging
import datetime
from collections import OrderedDict
from typing import Optional, Union, Dict

import torch
from tqdm import tqdm
from mmcv.runner import HOOKS
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.hooks.logger.text import TextLoggerHook


@HOOKS.register_module(force=True)
class TextLoggerHook(TextLoggerHook):

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
                         file_client_args)
        self.ndigits = ndigits
        self.handltype = []
        self.bar = None
        self.reset = True
        self.log_dict = None

    def before_train_epoch(self, runner):
        super().before_train_epoch(runner)
        self.reset = True

    def before_train_iter(self, runner):
        super().before_train_iter(runner)
        if self.log_dict:
            self._progress_log(self.log_dict, runner)

    def after_train_epoch(self, runner) -> None:
        super().after_train_epoch(runner)
        self.bar = None

    def _progress_log(self, log_dict, runner: BaseRunner):
        head = '\n'
        end = ''
        for key, value in log_dict.items():
            if 'time' in key or 'grad_norm' in key: continue
            head += f'{key:^10}'
            end += f'{self._round_float(value):^10}' if isinstance(
                value, float) else f'{str(value):^10}'
        if self.reset:
            print(head)
            self.bar = tqdm(total=len(runner.data_loader),
                            ncols=120,
                            leave=True)
            self.reset = False

        self.bar.set_description(end)
        self.bar.update(1)
        if self.bar.n == len(runner.data_loader):
            del self.bar
            print('\n')

    def _log_info(self, log_dict: Dict, runner) -> None:
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)  # type: ignore
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'  # type: ignore

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}]' \
                        f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (runner.iter -
                                                    self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, ' \
                        f'data_time: {log_dict["data_time"]:.3f}, '
                # statistic memory
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) ' \
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = f'{val:.{self.ndigits}f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)
        self.setloglevel(runner, logging.StreamHandler)
        runner.logger.info(log_str)

    def log(self, runner) -> OrderedDict:
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

    def setloglevel(self,
                    runner: BaseRunner,
                    handler: logging.Handler = logging.StreamHandler,
                    level: int = logging.ERROR):
        if handler in self.handltype: return
        for i, hand in enumerate(runner.logger.handlers):
            if type(hand) is handler:
                hand.setLevel(level)
                runner.logger.handlers[i] = hand
                self.handltype.append(type(hand))

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, self.ndigits)
        else:
            return items