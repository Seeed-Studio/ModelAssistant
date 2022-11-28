import os
import cv2
import numpy as np


def image_dataset_preprocess(path, img_size):
    """Calculate mean and std of images representative dataset,
        only calculate the previous 100 data."""
    assert os.path.exists(path), f'audio path {path} not exist, please check!'
    datas = []
    VALID_FORMAT = ['jpg', 'png', 'jpeg']
    for file in os.listdir(path):
        if file.split(".")[-1].lower() not in VALID_FORMAT:
            continue
        image = cv2.imread(os.path.join(path, file))
        image = cv2.resize(image, (img_size[1], img_size[0]))[:, :, ::-1]
        datas.append(image / 255.0)
    datas = np.array(datas)
    means, vars = [], []
    for i in range(datas.shape[-1]):
        means.append(np.mean(datas[:, :, :, i]))
        vars.append(np.var(datas[:, :, :, i]))

    return np.mean(datas), np.var(datas)

# if __name__ == '__main__':
#     # audio_dir = r'../datasets/yes/'
#     # mean, var = audio_dataset_preprocess(audio_dir, [8192])
#     images_dir = r'E:\datasets\coco128\images\train2017'
#     mean, var = image_dataset_preprocess(images_dir, [240, 240])
