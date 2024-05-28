# Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) OpenMMLab.

import os.path as osp
from os import PathLike
from typing import List, Tuple, Optional, Sequence, Union, Callable, Dict

import mmengine
import numpy as np
from mmengine.dataset import BaseDataset
from mmengine.fileio import (BaseStorageBackend, get_file_backend,
                             list_from_file)
from mmengine.logging import MMLogger
from typing import List, Optional, Sequence, Union
from sscma.registry import DATASETS, TRANSFORMS


def expanduser(path):
    """Expand ~ and ~user constructions.

    If user or $HOME is unknown, do nothing.
    """
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path
    

def find_folders(
    root: str,
    backend: Optional[BaseStorageBackend] = None
) -> Tuple[List[str], Dict[str, int]]:
    """Find classes by folders under a root."""
    # Pre-build file backend to prevent verbose file backend inference.
    backend = backend or get_file_backend(root, enable_singleton=True)
    folders = list(
        backend.list_dir_or_file(
            root,
            list_dir=True,
            list_file=False,
            recursive=False,
        ))
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folders, folder_to_idx


def get_samples(
    root: str,
    folder_to_idx: Dict[str, int],
    is_valid_file: Callable,
    backend: Optional[BaseStorageBackend] = None,
):
    """Make dataset by walking all images under a root."""
    samples = []
    available_classes = set()
    # Pre-build file backend to prevent verbose file backend inference.
    backend = backend or get_file_backend(root, enable_singleton=True)

    for folder_name in sorted(list(folder_to_idx.keys())):
        _dir = backend.join_path(root, folder_name)
        files = backend.list_dir_or_file(
            _dir,
            list_dir=False,
            list_file=True,
            recursive=True,
        )
        for file in sorted(list(files)):
            if is_valid_file(file):
                path = backend.join_path(folder_name, file)
                item = (path, folder_to_idx[folder_name])
                samples.append(item)
                available_classes.add(folder_name)

    empty_folders = set(folder_to_idx.keys()) - available_classes

    return samples, empty_folders
    

@DATASETS.register_module()
class CustomClsDataset(BaseDataset):
    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: Sequence = (),
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 classes: Union[str, Sequence[str], None] = None):
        assert (ann_file or data_prefix or data_root), \
            'One of `ann_file`, `data_root` and `data_prefix` must '\
            'be specified.'
        
        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))

        self.extensions = tuple(set([i.lower() for i in extensions]))

        ann_file = expanduser(ann_file)
        metainfo = self._compat_classes(metainfo, classes)

        transforms = []
        for transform in pipeline:
            if isinstance(transform, dict):
                transforms.append(TRANSFORMS.build(transform))
            else:
                transforms.append(transform)

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=transforms,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch)
        
        if not lazy_init:
            self.full_init()

    @property
    def img_prefix(self):
        return self.data_prefix['img_path']
    
    @property
    def CLASSES(self):
        return self._metainfo.get('classes', None)
    
    @property
    def class_to_idx(self):
        return {cat: i for i, cat in enumerate(self.CLASSES)}
    
    def get_gt_labels(self):
        gt_labels = np.array([self.get_data_info(i)['gt_label'] for i in range(len(self))])
        return gt_labels
    
    def get_cat_ids(self, idx: int) -> List[int]:
        return [int(self.get_data_info(idx)['gt_label'])]
    
    def _compat_classes(self, metainfo, classes):
        if isinstance(classes, str):
            class_names = mmengine.list_from_file(expanduser(classes))
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        elif classes is not None:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        if metainfo is None:
            metainfo = {}
        if classes is not None:
            metainfo = {'classes': tuple(class_names), **metainfo}
        
        return metainfo
    
    def _find_samples(self):
        classes, folder_to_idx = find_folders(self.img_prefix)
        samples, empty_classes = get_samples(self.img_prefix,
                                            folder_to_idx,
                                            self.is_valid_file)
        if len(samples) == 0:
            raise RuntimeError(
                f'Found 0 files in subfolders of: {self.data_prefix}. '
                f'Supported extensions are: {",".join(self.extensions)}')

        if self.CLASSES is not None:
            assert len(self.CLASSES) == len(classes), \
                f"The number of subfolders ({len(classes)}) doesn't match " \
                f'the number of specified classes ({len(self.CLASSES)}). ' \
                'Please check the data folder.'
        else:
            self._metainfo['classes'] = tuple(classes)
        
        if empty_classes:
            logger = MMLogger.get_current_instance()
            logger.warning(
                'Found no valid file in the folder '
                f'{", ".join(empty_classes)}. '
                f"Supported extensions are: {', '.join(self.extensions)}")

        self.folder_to_idx = folder_to_idx

        return samples
    
    def full_init(self):
        '''Convert to standard metainfo format'''
        super().full_init()

        if 'categories' in self._metainfo and 'classes' not in self._metainfo:

            categories = sorted(self._metainfo['categories'], key=lambda x: x['id'])

            self._metainfo['classes'] = tuple([cat['category_name'] for cat in categories])
        
    def __repr__(self):
        """Print the basic information of the dataset.

        Returns:
            str: Formatted string.
        """
        head = 'Dataset ' + self.__class__.__name__
        body = []
        if self._fully_initialized:
            body.append(f'Number of samples: \t{self.__len__()}')
        else:
            body.append("Haven't been initialized")

        if self.CLASSES is not None:
            body.append(f'Number of categories: \t{len(self.CLASSES)}')
        else:
            body.append('The `CLASSES` meta info is not set.')

        body.extend(self.extra_repr())

        if len(self.pipeline.transforms) > 0:
            body.append('With transforms:')
            for t in self.pipeline.transforms:
                body.append(f'    {t}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = []
        body.append(f'Annotation file: \t{self.ann_file}')
        body.append(f'Prefix of images: \t{self.img_prefix}')
        return body
    
    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)
    
    def load_data_list(self):
        if not self.ann_file:
            samples = self._find_samples()
        else:
            lines = list_from_file(self.ann_file)
            samples = [x.strip().rsplit(' ', 1) for x in lines]
        
        backend = get_file_backend(self.img_prefix, enable_singleton=True)
        data_list = []
        for filename, gt_label in samples:
            img_path = backend.join_path(self.img_prefix, filename)
            info = {'img_path': img_path, 'gt_label': int(gt_label)}
            data_list.append(info)

        return data_list    
    