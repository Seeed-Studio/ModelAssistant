import os

from mmdet.datasets.voc import VOCDataset
from mmdet.datasets.builder import DATASETS

from datasets.utils.download import check_file


@DATASETS.register_module()
class CustomVocdataset(VOCDataset):
    def __init__(self, **kwargs):
        kwargs['data_root']=os.path.join(check_file(kwargs['data_root'],data_name='voc'),'VOCdevkit','VOC2012')

        super(CustomVocdataset,self).__init__(**kwargs)