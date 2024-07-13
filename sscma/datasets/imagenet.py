# Copyright (c) OpenMMLab. All rights reserved.
from typing import List,Dict, Optional, Union,Sequence,Callable,Tuple
from os import PathLike

from mmengine import fileio
from mmengine.logging import MMLogger
from mmengine.dataset import BaseDataset
from mmengine.fileio import (BaseStorageBackend, get_file_backend,
                             list_from_file)

from mmengine.registry import DATASETS
import mmengine
import numpy as np
import os.path as osp


def expanduser(path):
    """Expand ~ and ~user constructions.

    If user or $HOME is unknown, do nothing.
    """
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path
    

def get_samples(
    root: str,
    folder_to_idx: Dict[str, int],
    is_valid_file: Callable,
    backend: Optional[BaseStorageBackend] = None,
):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        is_valid_file (Callable): A function that takes path of a file
            and check if the file is a valid sample file.
        backend (BaseStorageBackend | None): The file backend of the root.
            If None, auto infer backend from the root path. Defaults to None.

    Returns:
        Tuple[list, set]:

        - samples: a list of tuple where each element is (image, class_idx)
        - empty_folders: The folders don't have any valid files.
    """
    samples = []
    available_classes = set()
    # Pre-build file backend to prevent verbose file backend inference.
    backend = backend or get_file_backend(root, enable_singleton=True)

    if folder_to_idx is not None:
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
    else:
        files = backend.list_dir_or_file(
            root,
            list_dir=False,
            list_file=True,
            recursive=True,
        )
        samples = [file for file in sorted(list(files)) if is_valid_file(file)]
        empty_folders = None

    return samples, empty_folders

def find_folders(
    root: str,
    backend: Optional[BaseStorageBackend] = None
) -> Tuple[List[str], Dict[str, int]]:
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders
        backend (BaseStorageBackend | None): The file backend of the root.
            If None, auto infer backend from the root path. Defaults to None.

    Returns:
        Tuple[List[str], Dict[str, int]]:

        - folders: The name of sub folders under the root.
        - folder_to_idx: The map from folder name to class idx.
    """
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



@DATASETS.register_module()
class ImageNet(BaseDataset):
    """Base dataset for image classification task.

    This dataset support annotation file in `OpenMMLab 2.0 style annotation
    format`.

    .. _OpenMMLab 2.0 style annotation format:
        https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md

    Comparing with the :class:`mmengine.BaseDataset`, this class implemented
    several useful methods.
            
	A generic dataset for multiple tasks.

    The dataset supports two kinds of style.

    1. Use an annotation file to specify all samples, and each line indicates a
       sample(todo):

       The annotation file (for ``with_label=True``, supervised tasks.): ::

           folder_1/xxx.png 0
           folder_1/xxy.png 1
           123.png 4
           nsdf3.png 3
           ...

       The annotation file (for ``with_label=False``, unsupervised tasks.): ::

           folder_1/xxx.png
           folder_1/xxy.png
           123.png
           nsdf3.png
           ...

       Sample files: ::

           data_prefix/
           ├── folder_1
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           ├── 123.png
           ├── nsdf3.png
           └── ...

       Please use the argument ``metainfo`` to specify extra information for
       the task, like ``{'classes': ('bird', 'cat', 'deer', 'dog', 'frog')}``.

    2. Place all samples in one folder as below:

       Sample files (for ``with_label=True``, supervised tasks, we use the name
       of sub-folders as the categories names): ::

           data_prefix/
           ├── class_x
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           │       └── xxz.png
           └── class_y
               ├── 123.png
               ├── nsdf3.png
               ├── ...
               └── asd932_.png

       Sample files (for ``with_label=False``, unsupervised tasks, we use all
       sample files under the specified folder): ::

           data_prefix/
           ├── folder_1
           │   ├── xxx.png
           │   ├── xxy.png
           │   └── ...
           ├── 123.png
           ├── nsdf3.png
           └── ...

    If the ``ann_file`` is specified, the dataset will be generated by the
    first way, otherwise, try the second way.
    eg:
    `ImageNet <http://www.image-net.org>`_ Dataset.

    The dataset supports two kinds of directory format,

    ::

        imagenet
        ├── train
        │   ├──class_x
        |   |   ├── x1.jpg
        |   |   ├── x2.jpg
        |   |   └── ...
        │   ├── class_y
        |   |   ├── y1.jpg
        |   |   ├── y2.jpg
        |   |   └── ...
        |   └── ...
        ├── val
        │   ├──class_x
        |   |   └── ...
        │   ├── class_y
        |   |   └── ...
        |   └── ...
        └── test
            ├── test1.jpg
            ├── test2.jpg
            └── ...

    or ::

        imagenet
        ├── train
        │   ├── x1.jpg
        │   ├── y1.jpg
        │   └── ...
        ├── val
        │   ├── x3.jpg
        │   ├── y3.jpg
        │   └── ...
        ├── test
        │   ├── test1.jpg
        │   ├── test2.jpg
        │   └── ...
        └── meta
            ├── train.txt
            └── val.txt
    Examples:
        >>> from sscma.datasets import ImageNet
        >>> train_dataset = ImageNet(data_root='/dataset/imagenet100', split='train')
        >>> train_dataset
        Dataset ImageNet
            Number of samples:  1281167
            Number of categories:       1000
            Root of dataset:    data/imagenet
        >>> test_dataset = ImageNet(data_root='/dataset/imagenet100', split='val')
        >>> test_dataset
        Dataset ImageNet
            Number of samples:  50000
            Number of categories:       1000
            Root of dataset:    data/imagenet
            
    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (str | dict): Prefix for training data. Defaults to ''.
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        multi_label (bool): Not implement by now. Use multi label or not.
            Defaults to False.
        **kwargs: Other keyword arguments in : class:`BaseDataset`.

    """  # noqa: E501
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    
    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 **kwargs):
        
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}
        
        if split:
            splits = ['train', 'val', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"

            if split == 'test':
                logger = MMLogger.get_current_instance()
                logger.info(
                    'Since the ImageNet1k test set does not provide label'
                    'annotations, `with_label` is set to False')
                kwargs['with_label'] = False

            data_prefix = split if data_prefix == '' else data_prefix

            if ann_file == '':
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')


                    
        metainfo = self._compat_classes(metainfo, None)
        
        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.with_label = True

        if isinstance(data_prefix, str):
            data_prefix = dict(img_path=expanduser(data_prefix))
            
                
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=None,
            indices=None,
            serialize_data=True,
            pipeline=(),
            test_mode=False,
            lazy_init=False,
            max_refetch=1000)

    @property
    def img_prefix(self):
        """The prefix of images."""
        return self.data_prefix['img_path']

    @property
    def CLASSES(self):
        """Return all categories names."""
        return self._metainfo.get('classes', None)

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {cat: i for i, cat in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        gt_labels = np.array(
            [self.get_data_info(i)['gt_label'] for i in range(len(self))])
        return gt_labels

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            cat_ids (List[int]): Image category of specified index.
        """

        return [int(self.get_data_info(idx)['gt_label'])]

    def _compat_classes(self, metainfo, classes):
        """Merge the old style ``classes`` arguments to ``metainfo``."""
        if isinstance(classes, str):
            # take it as a file path
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

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True."""
        super().full_init()

        #  To support the standard OpenMMLab 2.0 annotation format. Generate
        #  metainfo in internal format from standard metainfo format.
        if 'categories' in self._metainfo and 'classes' not in self._metainfo:
            categories = sorted(
                self._metainfo['categories'], key=lambda x: x['id'])
            self._metainfo['classes'] = tuple(
                [cat['category_name'] for cat in categories])
            

    def _find_samples(self):
        """find samples from ``data_prefix``."""
        if self.with_label:
            classes, folder_to_idx = find_folders(self.img_prefix)
            samples, empty_classes = get_samples(
                self.img_prefix,
                folder_to_idx,
                is_valid_file=self.is_valid_file,
            )

            self.folder_to_idx = folder_to_idx

            if self.CLASSES is not None:
                assert len(self.CLASSES) == len(classes), \
                    f"The number of subfolders ({len(classes)}) doesn't " \
                    f'match the number of specified classes ' \
                    f'({len(self.CLASSES)}). Please check the data folder.'
            else:
                self._metainfo['classes'] = tuple(classes)
        else:
            samples, empty_classes = get_samples(
                self.img_prefix,
                None,
                is_valid_file=self.is_valid_file,
            )

        if len(samples) == 0:
            raise RuntimeError(
                f'Found 0 files in subfolders of: {self.data_prefix}. '
                f'Supported extensions are: {",".join(self.extensions)}')

        if empty_classes:
            logger = MMLogger.get_current_instance()
            logger.warning(
                'Found no valid file in the folder '
                f'{", ".join(empty_classes)}. '
                f"Supported extensions are: {', '.join(self.extensions)}")

        return samples

    def load_data_list(self):
        """Load image paths and gt_labels."""
        if not self.ann_file:
            samples = self._find_samples()
        elif self.with_label:
            lines = list_from_file(self.ann_file)
            samples = [x.strip().rsplit(' ', 1) for x in lines]
        else:
            samples = list_from_file(self.ann_file)

        # Pre-build file backend to prevent verbose file backend inference.
        backend = get_file_backend(self.img_prefix, enable_singleton=True)
        data_list = []
        for sample in samples:
            if self.with_label:
                filename, gt_label = sample
                img_path = backend.join_path(self.img_prefix, filename)
                info = {'img_path': img_path, 'gt_label': int(gt_label)}
            else:
                img_path = backend.join_path(self.img_prefix, sample)
                info = {'img_path': img_path}
            data_list.append(info)
        return data_list

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)


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

        body.extend(self.extra_repr())

        if len(self.pipeline.transforms) > 0:
            body.append('With transforms:')
            for t in self.pipeline.transforms:
                body.append(f'    {t}')

        lines = [head] + [' ' * 4 + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body