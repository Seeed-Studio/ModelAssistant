# Copyright (c) Seeed Tech Ltd. All rights reserved.
from typing import Optional

import numpy as np
from mmengine import Config
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""
    return torch_dist.distributed_c10d._get_default_group()


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def parse_pose_metainfo(metainfo: dict):
    if 'from_file' in metainfo:
        cfg_file = metainfo['fromfile']
        metainfo = Config.fromfile(cfg_file).dataset_info

    assert 'dataset_name' in metainfo
    assert 'keypoint_info' in metainfo
    assert 'skeleton_info' in metainfo
    assert 'joint_weights' in metainfo
    assert 'sigmas' in metainfo

    parsed = dict(
        dataset_name=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed['dataset_name'] = metainfo['dataset_name']

    # parse keypoint information
    parsed['num_keypoints'] = len(metainfo['keypoint_info'])

    for kpt_id, kpt in metainfo['keypoint_info'].items():
        kpt_name = kpt['name']
        parsed['keypoint_id2name'][kpt_id] = kpt_name
        parsed['keypoint_name2id'][kpt_name] = kpt_id
        parsed['keypoint_colors'].append(kpt.get('color', [255, 128, 0]))

        kpt_type = kpt.get('type', '')
        if kpt_type == 'upper':
            parsed['upper_body_ids'].append(kpt_id)
        elif kpt_type == 'lower':
            parsed['lower_body_ids'].append(kpt_id)

        swap_kpt = kpt.get('swap', '')
        if swap_kpt == kpt_name or swap_kpt == '':
            parsed['flip_indices'].append(kpt_name)
        else:
            parsed['flip_indices'].append(swap_kpt)
            pair = (swap_kpt, kpt_name)
            if pair not in parsed['flip_pairs']:
                parsed['flip_pairs'].append(pair)

    # parse skeleton information
    parsed['num_skeleton_links'] = len(metainfo['skeleton_info'])
    for _, sk in metainfo['skeleton_info'].items():
        parsed['skeleton_links'].append(sk['link'])
        parsed['skeleton_link_colors'].append(sk.get('color', [96, 96, 255]))

    # parse extra information
    parsed['dataset_keypoint_weights'] = np.array(metainfo['joint_weights'], dtype=np.float32)
    parsed['sigmas'] = np.array(metainfo['sigmas'], dtype=np.float32)

    if 'stats_info' in metainfo:
        parsed['stats_info'] = {}
        for name, val in metainfo['stats_info'].items():
            parsed['stats_info'][name] = np.array(val, dtype=np.float32)

    # formatting
    def _map(src, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        else:
            return mapping[src]

    parsed['flip_pairs'] = _map(parsed['flip_pairs'], mapping=parsed['keypoint_name2id'])
    parsed['flip_indices'] = _map(parsed['flip_indices'], mapping=parsed['keypoint_name2id'])
    parsed['skeleton_links'] = _map(parsed['skeleton_links'], mapping=parsed['keypoint_name2id'])

    parsed['keypoint_colors'] = np.array(parsed['keypoint_colors'], dtype=np.uint8)
    parsed['skeleton_link_colors'] = np.array(parsed['skeleton_link_colors'], dtype=np.uint8)

    return parsed
