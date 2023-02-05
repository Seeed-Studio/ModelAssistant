# Edge Impulse ML Blocks

[Edge Impulse](https://www.edgeimpulse.com/) is the leading development platform for machine learning on edge devices. 

The models in EdgeLab support running on Edge Impulse specific information is available in the [edgelab-ei-ml-blocks](https://github.com/Seeed-Studio/edgelab-ei-ml-blocks).

The following is an example of how to run the EdgeLab model on Edge Impulse, using the `edgelab-fomo` model. 

## Running the pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker
1. Fetch edgelab-fomo ei-ml-blocks
    ```
    git clone https://github.com/Seeed-Studio/edgelab-ei-ml-blocks
    cd edgelab-ei-ml-blocks/edgelab-fomo
    ```
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
3. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16.0 or higher.
4. Create a new Edge Impulse project, and make sure the labeling method is set to 'Bounding boxes'.
    - Click `Create New Project`

    ![create-project-1](../../_static/imgs/ei-ml-blocks-create-project.png)
    - Enter basic project information.

    ![create-project-2](../../_static/imgs/ei-ml-blocks-create-project2.png)

5. Add and label some data.
![dataset](../../_static/imgs/ei-ml-blocks-dataset.png)
6. Under **Create impulse** set the image size to e.g. 160x160, 320x320 or 640x640, add an 'Image' DSP block and an 'Object Detection' learn block.
![dataset](../../_static/imgs/ei-ml-blocks-design.png)
7. Open a command prompt or terminal window.
8. Initialize the block:

    ```
    $ edge-impulse-blocks init
    # Answer the questions, select "Object Detection" for 'What type of data does this model operate on?' and "FOMO" for 'What's the last layer...'
    ```

9. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

10. Build the container:

    ```
    $ docker build -t edgelab-fomo .
    ```

11. Run the container to test the script (you don't need to rebuild the container if you make changes):

    ```
    $ docker run --shm-size=1024m --rm -v $PWD:/scripts edgelab-fomo --data-directory data/ --epochs 30 --learning-rate 0.00001 --out-directory out/
    ```

12. This creates a .tflite file in the 'out' directory.

```{note}
If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.
```

## Fetching new data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16 or higher.
2. Open a command prompt or terminal window.
3. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

## Pushing the block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block:

    ```
    $ edge-impulse-blocks push
    ```

2. The block is now available under any of your projects, via  **Create impulse > Add learning block > Object Detection (Images)**.
![object-detection](../../_static/imgs/ei-ml-blocks-obj-det.png)

3. Download block output
![dl](../../_static/imgs/ei-ml-blocks-dl.png)
