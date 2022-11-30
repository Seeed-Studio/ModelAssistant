# EAT

> [END-TO-END AUDIO STRIKES BACK: BOOSTING AUGMENTATIONS TOWARDS AN EFFICIENT AUDIO CLASSIFICATION NETWORK](https://arxiv.org/pdf/2204.11479.pdf)

## Abstract

While efficient architectures and a plethora of augmentations for end-to-end image classification tasks
have been suggested and heavily investigated, state-of-the-art techniques for audio classifications still
rely on numerous representations of the audio signal together with large architectures, fine-tuned from
large datasets. By utilizing the inherited lightweight nature of audio and novel audio augmentations,
we were able to present an efficient end-to-end (e2e) network with strong generalization ability.
Experiments on a variety of sound classification sets demonstrate the effectiveness and robustness
of our approach, by achieving state-of-the-art results in various settings. Public code is available at:
<https://github.com/Alibaba-MIIL/AudioClassification>.

<div align=center>
<img src="../../demo/EAT.png" width="250" height="280">
</div>

## Results and Models

### speechcommand(35)

|    Model   | Scale | Lr schd | Param(K) | Flops(M) | Top-1(%) | config | pth | onnx | ncnn |
| :--------: | :---: | :-----: | :------: | :------: | :------: | :----: | :-: | :--: | :--: |
|  ETA-tiny  |  8192 |  Poliy  |  23.95   |   0.53   |    86.2    | [config](./ali_classiyf_small_8k_8192.py)| [model](https://github.com/Seeed-Studio/edgelab/releases/download/model_zoo/ali_classiyf_small_8k_35_8192.pth) | [onnx](https://github.com/Seeed-Studio/edgelab/releases/download/model_zoo/ali_classiyf_small_8k_35_8192.onnx) | [ncnn](https://github.com/Seeed-Studio/edgelab/releases/download/model_zoo/ali_classiyf_small_8k_35_8192.zip) |

### speechcommand(4)

|    Model   | Scale | Lr schd | Param(K) | Flops(M) | Top-1(%) | config | pth | onnx | ncnn |
| :--------: | :---: | :-----: | :------: | :------: | :------: | :----: | :-: | :--: | :--: |
|  ETA-tiny  |  8192 |  Poliy  |  22.81   |   0.53   |    97    | [config](./ali_classiyf_small_8k_8192.py)| [model](https://github.com/Seeed-Studio/edgelab/releases/download/model_zoo/ali_classiyf_small_8k_4_8192.pth) | [onnx](https://github.com/Seeed-Studio/edgelab/releases/download/model_zoo/ali_classiyf_small_8k_4_8192.onnx) | [ncnn](https://github.com/Seeed-Studio/edgelab/releases/download/model_zoo/ali_classiyf_small_8k_4_8192.ncnn.zip) |

## Citation

```latex
@article{Gazneli2022EAT,
   title={End-to-End Audio Strikes Back: Boosting Augmentations Towards An Efficient Audio Classification Network},
   author={Avi Gazneli, Gadi Zimerman, Tal Ridnik, Gilad Sharir, Asaf Noy},
   journal={arXiv preprint arXiv:2204.11479,},
   year={2022},
}
```
