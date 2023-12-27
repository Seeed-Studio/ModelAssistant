# Edge Impulse ML Blocks

[Edge Impulse](https://www.edgeimpulse.com/) is the leading development platform for machine learning on edge devices.

The models in [SSCMA](https://github.com/Seeed-Studio/SSCMA) support running on Edge Impulse specific information is available in the [sscma-ei-ml-blocks](https://github.com/Seeed-Studio/sscma-ei-ml-blocks). The following is an example of how to run the [SSCMA](https://github.com/Seeed-Studio/SSCMA) model on Edge Impulse, using the `sscma-fomo` model.

## Run the Pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Pipeline on Docker

01. Clone the sample repository.

    ```sh
    git clone https://github.com/Seeed-Studio/sscma-ei-ml-blocks && \
    cd sscma-ei-ml-blocks/sscma-fomo
    ```

02. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

03. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) `v1.16.0` or higher.

04. Create a new Edge Impulse project, and make sure the labeling method is set to `Bounding Boxes`.

    - Click **'Create New Project'** button.

      ![create-project-1](/static/ei/ei-ml-blocks-create-project.png)

    - Guess a project name and finish setup.

      ![create-project-2](/static/ei/ei-ml-blocks-create-project2.png)

05. Add labels and some data.

    ![dataset](/static/ei/ei-ml-blocks-dataset.png)

06. Under **Create Impulse** set the image size (e.g. `160x160`, `320x320` or `640x640`), add an `Image` DSP block and an `Object Detection` learn block.

    ![dataset](/static/ei/ei-ml-blocks-design.png)

07. Open a command prompt or terminal window.

08. Initialize the block.

    ```sh
    edge-impulse-blocks init # Answer the questions, select 'Object Detection' for 'What type of data does this model operate on?' and "FOMO" for 'What's the last layer...'
    ```

09. Fetch new data via.

    ```sh
    edge-impulse-blocks runner --download-data data/
    ```

10. Build the container.

    ```sh
    docker build -t sscma-fomo .
    ```

11. Run the container to test the script (you don't need to rebuild the container if you make changes).

    ```sh
    docker run --shm-size=1024m --rm -v $PWD:/scripts sscma-fomo --data-directory data/ --epochs 30 --learning-rate 0.00001 --out-directory out/
    ```

12. This creates a `.tflite` file in the `out` directory.

::: tip

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

:::

## Fetching the new Data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) `v1.16` or higher.

2. Open a command prompt or terminal window.

3. Fetch new data using the command below.

   ```sh
   edge-impulse-blocks runner --download-data data/
   ```

## Pushing the Block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block.

   ```sh
   edge-impulse-blocks push
   ```

2. The block is now available under any of your projects, via **Create impulse > Add learning block > Object Detection (Images)**.

   ![object-detection](/static/ei/ei-ml-blocks-obj-det.png)

3. Download block output.

   ![dl](/static/ei/ei-ml-blocks-dl.png)
