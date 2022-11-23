from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

from datasets.utils.download import check_file


@DATASETS.register_module()
class CustomCocoDataset(CocoDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 seg_suffix='.png',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):

        data_root = eval(data_root) if '[' in data_root else data_root
        data_root = check_file(data_root,
                               data_name="coco") if data_root else data_root
        super().__init__(ann_file, pipeline, classes, data_root, img_prefix,
                         seg_prefix, seg_suffix, proposal_file, test_mode,
                         filter_empty_gt, file_client_args)
