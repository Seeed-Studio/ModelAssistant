# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import json
from typing import Dict, Optional, Sequence, Union
from mmengine.hooks import Hook
from mmengine.runner import Runner

from sscma.registry import HOOKS
import numpy as np
from tqdm.std import tqdm

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class SemiHook(Hook):
    """ """

    def __init__(self, bure_epoch: Union[float, int] = 1) -> None:
        super().__init__()
        if isinstance(bure_epoch, float):
            assert (
                bure_epoch <= 1.0
            ), "The number of supervised training rounds must be less than the maximum number of rounds"

        self.bure_epoch = bure_epoch

    def before_run(self, runner: Runner) -> None:
        if isinstance(self.bure_epoch, float):
            self.bure_epoch = int(runner.max_epochs * self.bure_epoch)

        assert self.bure_epoch <= runner.max_epochs

    def before_train_epoch(self, runner: Runner) -> None:
        if self.bure_epoch == runner.epoch:
            # dataloader starts loading unlabeled dataset for semi-supervised training
            runner.train_dataloader.sampler.all_data = True


@HOOKS.register_module()
class LabelMatchHook(Hook):
    def __init__(
        self,
        bure_epoch=0,
        interval=10,
        by_epoch=False,
        percent=0.2,
        ann_file: Optional[str] = None,
        act_eval: bool = False,
    ) -> None:
        super().__init__()
        self.by_epoch = by_epoch
        self.interval = interval
        self.percent = percent
        self.bure_epoch = bure_epoch
        self.computer_thr = False
        self.posetive = 1000
        self.ann_file = ann_file
        self.act_eval = act_eval

    def before_run(self, runner: Runner) -> None:
        # computer bure epoch
        if isinstance(self.bure_epoch, float):
            self.bure_epoch = int(runner.max_epochs * self.bure_epoch)

        assert self.bure_epoch <= runner.max_epochs
        # get label dataset classes
        self.CLASSES: Union[tuple, list] = runner.train_dataloader.dataset.CLASSES
        self.score_list = [[] for _ in range(len(self.CLASSES))]
        # set ann file
        if self.ann_file is None:
            self.ann_file = runner.train_dataloader.dataset.sup_dataset.ann_file
        # get label dataset distribution info
        self.boxes_per_image_gt, self.cls_taio_gt = self.get_sup_distribution_info(self.ann_file)
        # computer perclass postive number for unlabel dataset
        self.per_cls_postive = (
            self.boxes_per_image_gt * self.cls_taio_gt * len(runner.train_dataloader.dataset.unsup_dataset)
        )

    def before_train_epoch(self, runner: Runner) -> None:
        if self.bure_epoch == runner.epoch:
            # dataloader starts loading unlabeled dataset for semi-supervised training
            runner.train_dataloader.sampler.all_data = True

    def after_train_epoch(self, runner: Runner) -> None:
        if not self.act_eval and self.every_n_epochs(runner, self.interval):
            runner.train_dataloader.sampler.only_unlabel = True
            self.test_dataset_result(runner.model, runner.train_dataloader)
            self.parse_thr(runner)
            runner.train_dataloader.sampler.all_data = True

        return super().after_train_epoch(runner)

    def get_sup_distribution_info(self, ann_file):
        # computer sup dataset label distribudion info
        with open(ann_file, 'r') as f:
            data_info = json.load(f)
        image_num = len(data_info['images'])
        cls_num = [0] * len(self.CLASSES)
        catid2index = {}
        for value in data_info['categories']:
            catid2index[value['id']] = self.CLASSES.index(value['name'])

        for value in data_info['annotations']:
            cls_num[catid2index[value['category_id']]] += 1
        total_boxes = sum(cls_num)
        per_cls_gt_tatio = np.asarray([num / total_boxes for num in cls_num])

        per_image_gt_boxes = sum(cls_num) / image_num

        return per_image_gt_boxes, per_cls_gt_tatio

    def test_dataset_result(self, model, dataload):
        self.score_list = [[] for _ in range(len(self.CLASSES))]
        for data in tqdm(dataload, desc="computer thr", ncols=80):
            batch_results = model.val_step(
                {"inputs": data['inputs']['unsup_teacher'], 'data_samples': data['data_samples']['unsup_teacher']}
            )
        self.parse_val_result(batch_results)

    def parse_val_result(self, results):
        for out in results:
            for idx, label in enumerate(out.pred_instances.labels):
                self.score_list[int(label)].append(float(out.pred_instances.scores[idx]))

    def parse_thr(self, runner: Runner):
        self.score_list = [sorted(cl, reverse=True) if len(cl) else [0] for cl in self.score_list]
        cls_thr = [0] * len(runner.train_dataloader.dataset.CLASSES)
        cls_thr_ig = [0] * len(runner.train_dataloader.dataset.CLASSES)
        for idx, scores in enumerate(self.score_list):
            cls_thr[idx] = max(0.05, scores[min(len(scores) - 1, int(self.per_cls_postive[idx] * self.percent))])
            cls_thr_ig[idx] = max(0.05, scores[min(len(scores) - 1, int(self.per_cls_postive[idx]))])

        runner.model.pseudo_label_creator.cls_thr = cls_thr
        runner.model.pseudo_label_creator.cls_thr_ig = cls_thr_ig

        self.score_list = [i.clear() for i in self.score_list]

    def before_val_epoch(self, runner: Runner) -> None:
        if self.act_eval and self.every_n_epochs(runner, self.interval):
            self.computer_thr = True

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        if self.computer_thr:
            self.computer_thr = False
            self.parse_thr(runner)

    def after_val_iter(
        self, runner: Runner, batch_idx: int, data_batch: DATA_BATCH = None, outputs: Optional[Sequence] = None
    ) -> None:
        if self.computer_thr:
            self.parse_val_result(outputs)

        return super().after_val_iter(runner, batch_idx, data_batch, outputs)

    def every_n_epochs(self, runner, n: int) -> bool:
        """Test whether current epoch can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            n (int): Whether current epoch can be evenly divided by n.

        Returns:
            bool: Whether current epoch can be evenly divided by n.
        """
        if runner.epoch + 1 <= self.bure_epoch:
            return False
        return (runner.epoch + 1 - self.bure_epoch) % n == 0 if n > 0 else False

    def every_n_train_iters(self, runner: Runner, n: int) -> bool:
        """Test whether current training iteration can be evenly divided by n.

        Args:
            runner (Runner): The runner of the training, validation or testing
                process.
            n (int): Whether current iteration can be evenly divided by n.

        Returns:
            bool: Return True if the current iteration can be evenly divided
            by n, otherwise False.
        """
        if runner.epoch + 1 <= self.bure_epoch:
            return False
        return (runner.iter + 1 - (len(runner.train_dataloader) * self.bure_epoch)) % n == 0 if n > 0 else False
