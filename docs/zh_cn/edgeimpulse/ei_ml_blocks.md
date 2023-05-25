# Edge Impulse ML Blocks

[Edge Impulse](https://www.edgeimpulse.com/) 是在边缘设备上进行机器学习的领先开发平台。

EdgeLab中的模型支持在Edge Impulse上运行，具体信息可在[edgelab-ei-ml-blocks](https://github.com/Seeed-Studio/edgelab-ei-ml-blocks)中查阅.

下面以 `edgelab-fomo` 模型为例，说明如何在Edge Impulse上运行EdgeLab模型。

## 运行步骤

通过Docker运行。这为你封装了所有的依赖性和包。

### Running via Docker
1. 获取 edgelab-fomo ei-ml-blocks
    ```
    git clone https://github.com/Seeed-Studio/edgelab-ei-ml-blocks
    cd edgelab-ei-ml-blocks/edgelab-fomo
    ```
2. 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/).
3. 安装 [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16.0 或以上版本.
4. 创建一个新的Edge Impulse项目，并确保标签方法被设置为 'Bounding boxes'.
    - 点击 `Create New Project`

    ![create-project-1](/static/ei/ei-ml-blocks-create-project.png)
    - 键入项目基本信息.

    ![create-project-2](/static/ei/ei-ml-blocks-create-project2.png)

5. 添加并标注一些数据.
![dataset](/static/ei/ei-ml-blocks-dataset.png)
6. 在 **Create impulse** 将图像大小设置为例如160x160、320x320或640x640，添加一个 "图像 "DSP块和一个 "物体检测 "学习块。
![dataset](/static/ei/ei-ml-blocks-design.png)
7. 打开一个命令提示符或终端窗口。.
8. 初始化ei-ml-blocks:

    ```
    $ edge-impulse-blocks init
    # Answer the questions, select "Object Detection" for 'What type of data does this model operate on?' and "FOMO" for 'What's the last layer...'
    ```

9. 通过以下方式获取新数据:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

10. 构建容器:

    ```
    $ docker build -t edgelab-fomo .
    ```

11. 运行容器来测试脚本（如果你做了修改，你不需要重建容器）。

    ```
    $ docker run --shm-size=1024m --rm -v $PWD:/scripts edgelab-fomo --data-directory data/ --epochs 30 --learning-rate 0.00001 --out-directory out/
    ```

12. 这将在'out'目录下创建一个.tflite文件。

```{note}
如果你有额外的软件包想在容器中安装，请将它们添加到`requirements.txt`中，然后重建容器。
```
## 获取新数据

从你的项目中获得最新的数据。

1. 安装 [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16 或以上版本.
2. 打开一个命令提示符或终端窗口.
3. 通过以下方式获取新数据:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

## 将 block 推送到  Edge Impulse

你也可以把这个block推回给Edge Impulse，这使得它像其他ML块一样可用，这样你就可以在新数据到来时重新训练你的模型，或者把模型部署到设备上。更多信息请参见 [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) 。

1. 推送 block:

    ```
    $ edge-impulse-blocks push
    ```

2. 该block现在可以在你的任何项目下使用，通过  **Create impulse > Add learning block > Object Detection (Images)**.
![object-detection](/static/ei/ei-ml-blocks-obj-det.png)

3. 下载 block 输出
![dl](/static/ei/ei-ml-blocks-dl.png)
