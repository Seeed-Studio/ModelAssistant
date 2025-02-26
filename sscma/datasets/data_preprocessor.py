# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Callable,Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numbers import Number
from typing import List, Optional, Sequence, Union


from mmengine.utils import is_seq_of
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor, stack_batch
from mmengine.registry import TRANSFORMS, MODELS
from mmengine import MessageHub


from mmengine.structures import PixelData
from sscma.structures import (
    DataSample,
    MultiTaskDataSample,
    DetDataSample,
    batch_label_to_onehot,
    cat_batch_labels,
    tensor_split,
)
from sscma.utils.misc import samplelist_boxtype2tensor

class BatchSyncRandomResize(nn.Module):
    """Batch random resize which synchronizes the random size across ranks.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def __init__(self,
                 random_size_range: Tuple[int, int],
                 interval: int = 10,
                 size_divisor: int = 32) -> None:
        super().__init__()
        self.rank, self.world_size = get_dist_info()
        self._input_size = None
        self._random_size_range = (round(random_size_range[0] / size_divisor),
                                   round(random_size_range[1] / size_divisor))
        self._interval = interval
        self._size_divisor = size_divisor

    def forward(
        self, inputs: torch.Tensor, data_samples: List[DetDataSample]
    ) -> Tuple[torch.Tensor, List[DetDataSample]]:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(
                inputs,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for data_sample in data_samples:
                img_shape = (int(data_sample.img_shape[0] * scale_y),
                             int(data_sample.img_shape[1] * scale_x))
                pad_shape = (int(data_sample.pad_shape[0] * scale_y),
                             int(data_sample.pad_shape[1] * scale_x))
                data_sample.set_metainfo({
                    'img_shape': img_shape,
                    'pad_shape': pad_shape,
                    'batch_input_shape': self._input_size
                })
                data_sample.gt_instances.bboxes[
                    ...,
                    0::2] = data_sample.gt_instances.bboxes[...,
                                                            0::2] * scale_x
                data_sample.gt_instances.bboxes[
                    ...,
                    1::2] = data_sample.gt_instances.bboxes[...,
                                                            1::2] * scale_y
                if 'ignored_instances' in data_sample:
                    data_sample.ignored_instances.bboxes[
                        ..., 0::2] = data_sample.ignored_instances.bboxes[
                            ..., 0::2] * scale_x
                    data_sample.ignored_instances.bboxes[
                        ..., 1::2] = data_sample.ignored_instances.bboxes[
                            ..., 1::2] * scale_y
        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(
                aspect_ratio=float(w / h), device=inputs.device)
        return inputs, data_samples

    def _get_random_size(self, aspect_ratio: float,
                         device: torch.device) -> Tuple[int, int]:
        """Randomly generate a shape in ``_random_size_range`` and broadcast to
        all ranks."""
        tensor = torch.LongTensor(2).to(device)
        if self.rank == 0:
            size = random.randint(*self._random_size_range)
            size = (self._size_divisor * size,
                    self._size_divisor * int(aspect_ratio * size))
            tensor[0] = size[0]
            tensor[1] = size[1]
        barrier()
        broadcast(tensor, 0)
        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
    


class YOLOXBatchSyncRandomResize(BatchSyncRandomResize):
    """YOLOX batch random resize.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def forward(self, inputs: torch.Tensor, data_samples: dict) -> torch.Tensor and dict: # type: ignore
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        inputs = inputs.float()
        assert isinstance(data_samples, dict)

        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(
                inputs,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)

            data_samples['bboxes_labels'][:, 2::2] *= scale_x
            data_samples['bboxes_labels'][:, 3::2] *= scale_y

            if 'keypoints' in data_samples:
                data_samples['keypoints'][..., 0] *= scale_x
                data_samples['keypoints'][..., 1] *= scale_y

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(
                aspect_ratio=float(w / h), device=inputs.device)

        return inputs, data_samples
    


class RandomBatchAugment:
    """Randomly choose one batch augmentation to apply.

    Args:
        augments (Callable | dict | list): configs of batch
            augmentations.
        probs (float | List[float] | None): The probabilities of each batch
            augmentations. If None, choose evenly. Defaults to None.

    Example:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from mmpretrain.models import RandomBatchAugment
        >>> augments_cfg = [
        ...     dict(type='CutMix', alpha=1.),
        ...     dict(type='Mixup', alpha=1.)
        ... ]
        >>> batch_augment = RandomBatchAugment(augments_cfg, probs=[0.5, 0.3])
        >>> imgs = torch.rand(16, 3, 32, 32)
        >>> label = F.one_hot(torch.randint(0, 10, (16, )), num_classes=10)
        >>> imgs, label = batch_augment(imgs, label)

    .. note ::

        To decide which batch augmentation will be used, it picks one of
        ``augments`` based on the probabilities. In the example above, the
        probability to use CutMix is 0.5, to use Mixup is 0.3, and to do
        nothing is 0.2.
    """

    def __init__(self, augments: Union[Callable, dict, list], probs=None):
        if not isinstance(augments, (tuple, list)):
            augments = [augments]

        self.augments = []
        for aug in augments:
            if isinstance(aug, dict):
                self.augments.append(TRANSFORMS.build(aug))
            else:
                self.augments.append(aug)

        if isinstance(probs, float):
            probs = [probs]

        if probs is not None:
            assert len(augments) == len(probs), (
                "``augments`` and ``probs`` must have same lengths. "
                f"Got {len(augments)} vs {len(probs)}."
            )
            assert sum(probs) <= 1, "The total probability of batch augments exceeds 1."
            self.augments.append(None)
            probs.append(1 - sum(probs))

        self.probs = probs

    def __call__(self, batch_input: torch.Tensor, batch_score: torch.Tensor):
        """Randomly apply batch augmentations to the batch inputs and batch
        data samples."""
        aug_index = np.random.choice(len(self.augments), p=self.probs)
        aug = self.augments[aug_index]

        if aug is not None:
            return aug(batch_input, batch_score)
        else:
            return batch_input, batch_score.float()


class ClsDataPreprocessor(BaseDataPreprocessor):
    """Image pre-processor for classification tasks.

    Comparing with the :class:`mmengine.model.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        to_onehot (bool): Whether to generate one-hot format gt-labels and set
            to data samples. Defaults to False.
        num_classes (int, optional): The number of classes. Defaults to None.
        batch_augments (dict, optional): The batch augmentations settings,
            including "augments" and "probs". For more details, see
            :class:`mmpretrain.models.RandomBatchAugment`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Number = 0,
        to_rgb: bool = False,
        to_onehot: bool = False,
        num_classes: Optional[int] = None,
        batch_augments: Optional[dict] = None,
    ):
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.to_onehot = to_onehot
        self.num_classes = num_classes

        if mean is not None:
            assert std is not None, (
                "To enable the normalization in "
                "preprocessing, please specify both `mean` and `std`."
            )
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        if batch_augments:
            self.batch_augments = RandomBatchAugment(**batch_augments)
            if not self.to_onehot:
                from mmengine.logging import MMLogger

                MMLogger.get_current_instance().info(
                    "Because batch augmentations are enabled, the data "
                    "preprocessor automatically enables the `to_onehot` "
                    "option to generate one-hot format labels."
                )
                self.to_onehot = True
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        inputs = self.cast_data(data["inputs"])

        if isinstance(inputs, torch.Tensor):
            # The branch if use `default_collate` as the collate_fn in the
            # dataloader.

            # ------ To RGB ------
            if self.to_rgb and inputs.size(1) == 3:
                inputs = inputs.flip(1)

            # -- Normalization ---
            inputs = inputs.float()
            if self._enable_normalize:
                inputs = (inputs - self.mean) / self.std

            # ------ Padding -----
            if self.pad_size_divisor > 1:
                h, w = inputs.shape[-2:]

                target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                inputs = F.pad(inputs, (0, pad_w, 0, pad_h), "constant", self.pad_value)
        else:
            # The branch if use `pseudo_collate` as the collate_fn in the
            # dataloader.

            processed_inputs = []
            for input_ in inputs:
                # ------ To RGB ------
                if self.to_rgb and input_.size(0) == 3:
                    input_ = input_.flip(0)

                # -- Normalization ---
                input_ = input_.float()
                if self._enable_normalize:
                    input_ = (input_ - self.mean) / self.std

                processed_inputs.append(input_)
            # Combine padding and stack
            inputs = stack_batch(
                processed_inputs, self.pad_size_divisor, self.pad_value
            )

        data_samples = data.get("data_samples", None)
        sample_item = data_samples[0] if data_samples is not None else None

        if isinstance(sample_item, DataSample):
            batch_label = None
            batch_score = None

            if "gt_label" in sample_item:
                gt_labels = [sample.gt_label for sample in data_samples]
                batch_label, label_indices = cat_batch_labels(gt_labels)
                batch_label = batch_label.to(self.device)
            if "gt_score" in sample_item:
                gt_scores = [sample.gt_score for sample in data_samples]
                batch_score = torch.stack(gt_scores).to(self.device)
            elif self.to_onehot and "gt_label" in sample_item:
                assert (
                    batch_label is not None
                ), "Cannot generate onehot format labels because no labels."
                num_classes = self.num_classes or sample_item.get("num_classes")
                assert num_classes is not None, (
                    "Cannot generate one-hot format labels because not set "
                    "`num_classes` in `data_preprocessor`."
                )
                batch_score = batch_label_to_onehot(
                    batch_label, label_indices, num_classes
                ).to(self.device)

            # ----- Batch Augmentations ----
            if training and self.batch_augments is not None and batch_score is not None:
                inputs, batch_score = self.batch_augments(inputs, batch_score)

            # ----- scatter labels and scores to data samples ---
            if batch_label is not None:
                for sample, label in zip(
                    data_samples, tensor_split(batch_label, label_indices)
                ):
                    sample.set_gt_label(label)
            if batch_score is not None:
                for sample, score in zip(data_samples, batch_score):
                    sample.set_gt_score(score)
        elif isinstance(sample_item, MultiTaskDataSample):
            data_samples = self.cast_data(data_samples)

        return {"inputs": inputs, "data_samples": data_samples}


class DetDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It supports batch augmentations.
    2. It will additionally append batch_input_shape and pad_shape
    to data_samples considering the object detection task.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        boxtype2tensor: bool = True,
        non_blocking: Optional[bool] = True,
        batch_augments: Optional[List[dict]] = None,
    ):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking,
        )
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments]
            )
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor

    def forward(self, data, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            batch_pad_shape = self._get_pad_shape(data)
            data = super().forward(data=data, training=training)
            inputs, data_samples = data['inputs'], data['data_samples']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which is
                # then used for the transformer_head.
                batch_input_shape = tuple(inputs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if self.boxtype2tensor:
                    samplelist_boxtype2tensor(data_samples)

                if self.pad_mask:
                    self.pad_gt_masks(data_samples)

                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    inputs, data_samples = batch_aug(inputs, data_samples)

            return {'inputs': inputs, 'data_samples': data_samples}


        data = self.cast_data(data)
        inputs, data_samples = data["inputs"], data["data_samples"]
        assert isinstance(data["data_samples"], dict)

        # TODO: Supports multi-scale training
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        img_metas = [{"batch_input_shape": inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            "bboxes_labels": data_samples["bboxes_labels"],
            "img_metas": img_metas,
        }
        if "masks" in data_samples:
            data_samples_output["masks"] = data_samples["masks"]
        if "keypoints" in data_samples:
            data_samples_output["keypoints"] = data_samples["keypoints"]
            data_samples_output["keypoints_visible"] = data_samples["keypoints_visible"]

        return {"inputs": inputs, "data_samples": data_samples_output}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data["inputs"]
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = (
                    int(np.ceil(ori_input.shape[1] / self.pad_size_divisor))
                    * self.pad_size_divisor
                )
                pad_w = (
                    int(np.ceil(ori_input.shape[2] / self.pad_size_divisor))
                    * self.pad_size_divisor
                )
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                "The input of `ImgDataPreprocessor` should be a NCHW tensor "
                "or a list of tensor, but got a tensor with shape: "
                f"{_batch_inputs.shape}"
            )
            pad_h = (
                int(np.ceil(_batch_inputs.shape[2] / self.pad_size_divisor))
                * self.pad_size_divisor
            )
            pad_w = (
                int(np.ceil(_batch_inputs.shape[3] / self.pad_size_divisor))
                * self.pad_size_divisor
            )
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError(
                "Output of `cast_data` should be a dict "
                "or a tuple with inputs and data_samples, but got"
                f"{type(data)}: {data}"
            )
        return batch_pad_shape

    def pad_gt_masks(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if "masks" in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape, pad_val=self.mask_pad_value
                )

    def pad_gt_sem_seg(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if "gt_sem_seg" in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode="constant",
                    value=self.seg_pad_value,
                )
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)
