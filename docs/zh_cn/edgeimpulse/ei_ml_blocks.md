# Edge Impulse 机器学习块

[Edge Impulse](https://www.edgeimpulse.com/) 是边缘设备上机器学习的领先开发平台。

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 中的模型支持在 Edge Impulse 上运行，具体信息见 GitHub 仓库 [sscma-ei-ml-blocks](https://github.com/Seeed-Studio/sscma-ei-ml-blocks)。下面使用 `sscma-fomo` 模型进行示例，说明如何在 Edge Impulse 上运行 SSCMA 模型。

## 运行管线

我们的示例通过 Docker 运行整个部署流程，这为你封装了所有的依赖和包。

### 管线在 Docker 上

01. 克隆示例仓库。

    ```sh
    git clone https://github.com/Seeed-Studio/sscma-ei-ml-blocks && \
    cd sscma-ei-ml-blocks/sscma-fomo
    ```

02. 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)。

03. 安装 [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) `v1.16.0` 或更高版本。

04. 创建一个新的Edge Impulse项目，并确保标签方法被设置为 "Bounding Boxes"。

    - 点击"创建新项目"按钮。

      ![create-project-1](https://files.seeedstudio.com/sscma/docs/static/ei/ei-ml-blocks-create-project.png)

    - 思考一个项目名称并完成设置。

      ![create-project-2](https://files.seeedstudio.com/sscma/docs/static/ei/ei-ml-blocks-create-project2.png)

05. 添加标签和一些数据。

    ![dataset](https://files.seeedstudio.com/sscma/docs/static/ei/ei-ml-blocks-dataset.png)

06. 在 **Create Impulse** 下设置图像大小 (例如:`160x160`，`320x320`或`640x640`)，添加一个`图像` DSP 块和一个`物体检测`学习块。

    ![dataset](https://files.seeedstudio.com/sscma/docs/static/ei/ei-ml-blocks-design.png)

07. 打开一个命令提示符或终端窗口。

08. 初始化该块。

    ```sh
    edge-impulse-blocks init # 回答问题，在 "这个模型对什么类型的数据进行操作？" 中选择 "Object Detection"，在 "最后一层是什么..." 中选择 "FOMO"
    ```

09. 通过以下方式获取新数据。

    ```sh
    edge-impulse-blocks runner --download-data data/
    ```

10. 构建容器。

    ```sh
    docker build -t sscma-fomo .
    ```

11. 运行容器来测试脚本 (如果你做了修改，你不需要重建容器)。

    ```sh
    docker run --shm-size=1024m --rm -v $PWD:/scripts sscma-fomo --data-directory data/ --epochs 30 --learning-rate 0.00001 --out-directory out/.
    ```

12. 这将在 `out` 目录下创建一个 `.tflite` 文件。

::: tip

如果你有额外的软件包想在容器内安装，把它们添加到 `requirements.txt` 中，然后重建容器。

:::

## 获取新数据

要从你的项目中获取最新的数据:

1. 安装 [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) `v1.16` 或更高版本。

2. 打开一个命令提示符或终端窗口。

3. 使用下面的命令获取新数据。

   ```sh
   edge-impulse-blocks runner --download-data data/
   ```

## 把块推回 Edge Impulse

你也可以把这个块推回 Edge Impulse，这使得它像其他 ML 块一样可用，这样你就可以在新数据进来的时候重新训练你的模型，或者把模型部署到设备上。更多信息请参见 [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models)。

1. 推送该块。

   ```sh
   edge-impulse-blocks push
   ```

2. 该块现在可以在你的任何项目下使用，通过 **Create impulse > Add learning block > Object Detection (Images)**。

   ![object-detection](https://files.seeedstudio.com/sscma/docs/static/ei/ei-ml-blocks-obj-det.png)

3. 下载块的输出。

   ![dl](https://files.seeedstudio.com/sscma/docs/static/ei/ei-ml-blocks-dl.png)
