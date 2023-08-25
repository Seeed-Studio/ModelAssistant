from typing import Optional

from mmcls.models.utils.batch_augments import RandomBatchAugment
from mmcls.structures import (
    ClsDataSample,
    MultiTaskDataSample,
    batch_label_to_onehot,
    cat_batch_labels,
    stack_batch_scores,
    tensor_split,
)
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor

from sscma.registry import MODELS


@MODELS.register_module()
class SensorDataPreprocessor(BaseDataPreprocessor):
    def __init__(
        self, to_onehot: bool = False, num_classes: Optional[int] = None, batch_augments: Optional[dict] = None
    ):
        super().__init__()
        self.to_onehot = to_onehot
        self.num_classes = num_classes

        if batch_augments is not None:
            self.batch_augments = RandomBatchAugment(**batch_augments)
            if not self.to_onehot:
                from mmengine.logging import MMLogger

                MMLogger.get_current_instance().info(
                    'Because batch augmentations are enabled, the data '
                    'preprocessor automatically enables the `to_onehot` '
                    'option to generate one-hot format labels.'
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
        inputs = self.cast_data(data['inputs'])

        data_samples = data.get('data_samples', None)

        sample_item = data_samples[0] if data_samples is not None else None
        if isinstance(sample_item, ClsDataSample) and 'gt_label' in sample_item:
            gt_labels = [sample.gt_label for sample in data_samples]
            batch_label, label_indices = cat_batch_labels(gt_labels, device=self.device)

            batch_score = stack_batch_scores(gt_labels, device=self.device)
            if batch_score is None and self.to_onehot:
                assert batch_label is not None, 'Cannot generate onehot format labels because no labels.'
                num_classes = self.num_classes or data_samples[0].get('num_classes')
                assert num_classes is not None, (
                    'Cannot generate one-hot format labels because not set ' '`num_classes` in `data_preprocessor`.'
                )
                batch_score = batch_label_to_onehot(batch_label, label_indices, num_classes)

            # ----- Batch Augmentations ----
            if training and self.batch_augments is not None:
                inputs, batch_score = self.batch_augments(inputs, batch_score)

            # ----- scatter labels and scores to data samples ---
            if batch_label is not None:
                for sample, label in zip(data_samples, tensor_split(batch_label, label_indices)):
                    sample.set_gt_label(label)
            if batch_score is not None:
                for sample, score in zip(data_samples, batch_score):
                    sample.set_gt_score(score)
        elif isinstance(sample_item, MultiTaskDataSample):
            data_samples = self.cast_data(data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}
