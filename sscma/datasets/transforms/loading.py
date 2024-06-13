# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import json
import warnings
from typing import Optional, List, Tuple

import torch
import numpy as np

import mmengine.fileio as fileio
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations as MMDET_LoadAnnotations
from mmdet.structures.bbox import get_box_type
from mmdet.structures.mask import PolygonMasks

from sscma.registry import TRANSFORMS
from .keypoint_structure import Keypoints


@TRANSFORMS.register_module()
class LoadSensorFromFile(BaseTransform):
    """Load an Sensor sample data from file.

    Required keys:

    - "file_path": Path to the Sensor sample data file.

    Modified keys:

    - "data": Sensor sample data loaded from the file.
    - "sensor": Sensor type and unit loaded from the file.
    """

    def __init__(self, file_client_args: Optional[dict] = None, backend_args: Optional[dict] = None) -> None:
        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. ' 'Please use "backend_args" instead',
                DeprecationWarning,
            )
            if backend_args is not None:
                raise ValueError('"file_client_args" and "backend_args" cannot be set ' 'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load Axes.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['file_path']

        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(self.file_client_args, filename)
                lable_bytes = file_client.get(filename)
            else:
                lable_bytes = fileio.get(filename, backend_args=self.backend_args)
            label = json.loads(lable_bytes)
            sensors = label['payload']['sensors']
            data = np.array([], np.float32)
            for value in label['payload']['values']:
                data = np.append(data, value)
        except Exception as e:
            raise e

        results['data'] = data
        results['sensors'] = sensors

        return results


@TRANSFORMS.register_module()
class YOLOLoadAnnotations(MMDET_LoadAnnotations):
    """Because the yolo series does not need to consider ignore bboxes for the
    time being, in order to speed up the pipeline, it can be excluded in
    advance.

    Args:
        mask2bbox (bool): Whether to use mask annotation to get bbox.
            Defaults to False.
        poly2mask (bool): Whether to transform the polygons to bitmaps.
            Defaults to False.
        merge_polygons (bool): Whether to merge polygons into one polygon.
            If merged, the storage structure is simpler and training is more
            effcient, especially if the mask inside a bbox is divided into
            multiple polygons. Defaults to True.
    """

    def __init__(self, mask2bbox: bool = False, poly2mask: bool = False, merge_polygons: bool = True, **kwargs):
        self.mask2bbox = mask2bbox
        self.merge_polygons = merge_polygons
        assert not poly2mask, 'Does not support BitmapMasks considering ' 'that bitmap consumes more memory.'
        super().__init__(poly2mask=poly2mask, **kwargs)
        if self.mask2bbox:
            assert self.with_mask, 'Using mask2bbox requires ' 'with_mask is True.'
        self._mask_ignore_flag = None

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        if self.mask2bbox:
            self._load_masks(results)
            if self.with_label:
                self._load_labels(results)
                self._update_mask_ignore_data(results)
            gt_bboxes = results['gt_masks'].get_bboxes(dst_type='hbox')
            results['gt_bboxes'] = gt_bboxes
        elif self.with_keypoints:
            self._load_kps(results)
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(results.get('bbox', []), dtype=torch.float32)
        else:
            results = super().transform(results)
            self._update_mask_ignore_data(results)
        return results

    def _update_mask_ignore_data(self, results: dict) -> None:
        if 'gt_masks' not in results:
            return

        if 'gt_bboxes_labels' in results and len(results['gt_bboxes_labels']) != len(results['gt_masks']):
            assert len(results['gt_bboxes_labels']) == len(self._mask_ignore_flag)
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][self._mask_ignore_flag]

        if 'gt_bboxes' in results and len(results['gt_bboxes']) != len(results['gt_masks']):
            assert len(results['gt_bboxes']) == len(self._mask_ignore_flag)
            results['gt_bboxes'] = results['gt_bboxes'][self._mask_ignore_flag]

    def _load_bboxes(self, results: dict):
        """Private function to load bounding box annotations.
        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes.append(instance['bbox'])
                gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        if self.box_type is None:
            results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)

    def _load_labels(self, results: dict):
        """Private function to load label annotations.

        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(gt_bboxes_labels, dtype=np.int64)

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_masks = []
        gt_ignore_flags = []
        self._mask_ignore_flag = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                if 'mask' in instance:
                    gt_mask = instance['mask']
                    if isinstance(gt_mask, list):
                        gt_mask = [
                            np.array(polygon) for polygon in gt_mask if len(polygon) % 2 == 0 and len(polygon) >= 6
                        ]
                        if len(gt_mask) == 0:
                            # ignore
                            self._mask_ignore_flag.append(0)
                        else:
                            if len(gt_mask) > 1 and self.merge_polygons:
                                gt_mask = self.merge_multi_segment(gt_mask)
                            gt_masks.append(gt_mask)
                            gt_ignore_flags.append(instance['ignore_flag'])
                            self._mask_ignore_flag.append(1)
                    else:
                        raise NotImplementedError('Only supports mask annotations in polygon ' 'format currently')
                else:
                    # TODO: Actually, gt with bbox and without mask needs
                    #  to be retained
                    self._mask_ignore_flag.append(0)
        self._mask_ignore_flag = np.array(self._mask_ignore_flag, dtype=bool)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        h, w = results['ori_shape']
        gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def merge_multi_segment(self, gt_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Merge multi segments to one list.

        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all
        segments into one.
        Args:
            gt_masks(List(np.array)):
                original segmentations in coco's json file.
                like [segmentation1, segmentation2,...],
                each segmentation is a list of coordinates.
        Return:
            gt_masks(List(np.array)): merged gt_masks
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in gt_masks]
        idx_list = [[] for _ in range(len(gt_masks))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = self.min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        # first round: first to end, i.e. A->B(partial)->C
        # second round: end to first, i.e. C->B(remaining)-A
        for k in range(2):
            # forward first round
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]
                    # add the idx[0] point for connect next segment
                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate([segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    # deal with the middle segment
                    # Note that in the first round, only partial segment
                    # are appended.
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0] : idx[1] + 1])
            # forward second round
            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    # deal with the middle segment
                    # append the remaining points
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return [
            np.concatenate(s).reshape(
                -1,
            )
        ]

    def min_index(self, arr1: np.ndarray, arr2: np.ndarray) -> Tuple[int, int]:
        """Find a pair of indexes with the shortest distance.

        Args:
            arr1: (N, 2).
            arr2: (M, 2).
        Return:
            tuple: a pair of indexes.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        results['height'] = results['img_shape'][0]
        results['width'] = results['img_shape'][1]
        num_instances = len(results.get('bbox', []))

        if num_instances == 0:
            results['keypoints'] = np.empty((0, len(results['flip_indices']), 2), dtype=np.float32)
            results['keypoints_visible'] = np.empty((0, len(results['flip_indices'])), dtype=np.int32)
            results['category_id'] = []

        results['gt_keypoints'] = Keypoints(
            keypoints=results['keypoints'],
            keypoints_visible=results['keypoints_visible'],
            flip_indices=results['flip_indices'],
        )

        results['gt_ignore_flags'] = np.array([False] * num_instances)
        results['gt_bboxes_labels'] = np.array(results['category_id']) - 1

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'mask2bbox={self.mask2bbox}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str
