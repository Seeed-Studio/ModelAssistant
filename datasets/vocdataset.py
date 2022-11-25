from mmdet.datasets.voc import VOCDataset
from mmdet.datasets.builder import DATASETS

from datasets.utils.download import check_file

@DATASETS.register_module()
class CustomVocdataset(VOCDataset):
    def __init__(self, **kwargs):
        kwargs['data_root']=check_file(kwargs['data_root'],data_name='voc')

        super(CustomVocdataset,self).__init__(**kwargs)