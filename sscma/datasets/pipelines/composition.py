# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from typing import Dict, Optional, Sequence, Union

import albumentations as A
from albumentations.core.bbox_utils import BboxParams
from albumentations.core.composition import BaseCompose, BasicTransform
from albumentations.core.keypoints_utils import KeypointParams

from mmengine import TRANSFORMS


class AlbCompose(A.Compose):
    """The packaging of the compose class of alb, the purpose is to parse the
    pipeline in the configuration file.

    Args:
        transforms(list):The packaging of the compose class of alb, the purpose is to
            parse the pipeline in the configuration file
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old
            target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
    """

    def __init__(
        self,
        transforms: Sequence[Dict],
        bbox_params: Optional[Union[dict, "BboxParams"]] = None,
        keypoint_params: Optional[Union[dict, "KeypointParams"]] = None,
        additional_targets: Optional[Dict[str, str]] = None,
        p: float = 1,
    ):
        pose_trans = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                pose_trans.append(transform)
            elif isinstance(transforms, (BaseCompose, BasicTransform)):
                pose_trans.append(transform)
            else:
                raise TypeError(
                    "transform must be callable or a dict, but got"
                    f" {type(transform)}"
                )
            if isinstance(keypoint_params, str):
                keypoint_params = A.KeypointParams(keypoint_params)

        super().__init__(
            transforms=pose_trans,
            bbox_params=bbox_params,
            keypoint_params=keypoint_params,
            additional_targets=additional_targets,
            p=p,
        )
