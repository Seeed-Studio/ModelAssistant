import logging
import datetime
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Union, Dict

import torch
import torch.distributed as dist
from tqdm import tqdm
from edgelab.registry import HOOKS
from mmengine.runner import Runner
from mmengine.hooks.logger_hook import LoggerHook
from mmengine.structures.base_data_element import BaseDataElement


@HOOKS.register_module(force=True)
class TextLoggerHook(LoggerHook):

    def __init__(self,
                 interval: int = 10,
                 ignore_last: bool = False,
                 interval_exp_name: int = 1000,
                 out_dir: Optional[Union[str, Path]] = None,
                 out_suffix=...,
                 keep_local: bool = True,
                 file_client_args: Optional[dict] = None,
                 log_metric_by_epoch: bool = True,
                 backend_args: Optional[dict] = None):
        super().__init__(interval, ignore_last, interval_exp_name, out_dir,
                         out_suffix, keep_local, file_client_args,
                         log_metric_by_epoch, backend_args)

        self.ndigits = 4
        self.handltype = []
        self.bar = None
        self.reset = True
        self.log_dict = None
        self.head = None
        self.currentIter = 0
        self.trainIdx = 0
        self.valIdx = 0
        self.logData = BaseDataElement()
        self.trainLogData = BaseDataElement()
        self.valLogData = BaseDataElement()
        self.testLogData = BaseDataElement()

    def before_run(self, runner) -> None:
        super().before_run(runner)
        self.setloglevel(runner, logging.StreamHandler)

    def _before_epoch(self, runner, mode: str = 'train') -> None:
        self.head = ''
        self.reset = True
        return super()._before_epoch(runner, mode)

    def before_train_epoch(self, runner):
        print('-' * 120)
        super().before_train_epoch(runner)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[Union[dict, tuple, list]] = None,
                         outputs: Optional[dict] = None) -> None:
        super().after_train_iter(runner, batch_idx, data_batch, outputs)
        self._progress_log(outputs,
                           runner,
                           runner.train_dataloader,
                           batch_idx,
                           mode='train')

    def after_val_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch=None,
                       outputs=None) -> None:
        super().after_val_iter(runner=runner,
                               batch_idx=batch_idx,
                               data_batch=data_batch,
                               outputs=outputs)
        parsed_cfg = runner.log_processor._parse_windows_size(
            runner, batch_idx, runner.log_processor.custom_cfg)
        log_tag = runner.log_processor._collect_scalars(
            parsed_cfg, runner, 'val')
        self._progress_log(log_tag,
                           runner,
                           runner.val_dataloader,
                           batch_idx,
                           mode='val')

    def after_train_epoch(self, runner) -> None:
        super().after_train_epoch(runner)
        self.bar = None
        print('')

    def _after_epoch(self, runner, mode: str = 'train') -> None:

        return super()._after_epoch(runner, mode)

    def _progress_log(self,
                      log_dict: dict,
                      runner: Runner,
                      dataloader,
                      idx: int,
                      mode='train'):
        head = '\n'
        end = ''
        current_epoch = runner.epoch
        max_epochs = runner.max_epochs

        if self.reset:
            self.logData.set_data({k: [] for k in log_dict.keys()})

        head += "Mode".center(10)
        end += f"{mode:^10}"
        head += "Epoch".center(10)
        end += f"{(current_epoch+1) if mode=='train' else current_epoch}/{max_epochs}".center(
            10)

        for key, value in log_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() > 1:
                    continue
                else:
                    value = value.cpu().item()
            if isinstance(value, (float, int)):
                self.logData.get(key).append(value)
                value = sum(self.logData.get(key)) / len(self.logData.get(key))

            head += f'{key:^10}'
            end += f'{self._round_float(value):^10}' if isinstance(
                value, float) else f'{value:^10}'

        eta_sec = runner.message_hub.get_info('eta')
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        head += f'{"eta":^10}'
        end += f'{eta_str:^10}'

        if head != self.head:
            self.head = head
        if self.reset:
            print(self.head)
            self.bar = tqdm(total=len(dataloader), ncols=120, leave=True)
            self.reset = False

        self.bar.set_description(end)

        # self.bar.update(runner.val_interval if mode == 'val' else 100)
        self.bar.update(self.progressInterval(idx, mode=mode))
        if self.bar.n == len(dataloader):
            del self.bar

    def setloglevel(self,
                    runner: Runner,
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

    def _get_max_memory(self, runner: Runner) -> int:
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([int(mem) // (1048576)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return f'{mem_mb.item()}MB'

    def progressInterval(self, idx: int, mode: str = 'train'):
        if mode == 'train':
            if idx < self.trainIdx:
                self.trainIdx = idx
                res = idx
            else:
                res = idx - self.trainIdx
                self.trainIdx = idx

        else:
            if idx < self.valIdx:
                self.valIdx = idx
                res = idx
            else:
                res = idx - self.valIdx
                self.valIdx = idx
        return res if res else 1

    def iterInterval(self, runner: Runner):
        interval = runner.iter - self.currentIter
        self.currentIter = runner.iter
        return interval