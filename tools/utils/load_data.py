import os
import os.path as osp

import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets.pipelines.pose_transform import Pose_Compose

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Load_data(Dataset):

    def __init__(self, path, pipline=None) -> None:
        super(Dataset, self).__init__()
        if pipline:
            self.pipline = Pose_Compose(pipline)

        self.test_trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        if isinstance(path, str):
            if osp.isdir(path):
                file_ls = os.listdir(path)
                self.file_ls = [
                    osp.join(path, i) for i in file_ls
                    if i.split('.')[-1].lower() in IMG_FORMATS
                ]
            else:
                self.file_ls = [path]

    def __len__(self):
        return len(self.file_ls)

    def __getitem__(self, index):
        img_file = self.file_ls[index]
        img = cv2.imread(img_file)
        img = self.pipline(image=img)['image']
        img = self.test_trans(img)
        return {'img': img, 'image_file': img_file}


def test_load(path, pipline):
    if isinstance(path, str):
        datasets = Load_data(path, pipline)
    else:
        raise TypeError(
            f'Please enter the string of the image path, currently obtained is {type(path)} type({path})'
        )

    dataload = DataLoader(datasets, )
    return dataload
