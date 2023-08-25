from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmcls.structures import ClsDataSample
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer

from sscma.registry import VISUALIZERS


@VISUALIZERS.register_module()
class FomoLocalVisualizer(DetLocalVisualizer):
    """Unified Fomo and target detection visualization classes."""

    def __init__(self, name='v', *args, fomo=False, **kwargs) -> None:
        print(args)
        print(kwargs)
        super().__init__(*args, name=name, **kwargs)
        self.fomo = fomo

    @master_only
    def add_datasample(self, *args, **kwargs):
        if self.fomo:
            self.fomo_add_datasample(*args, **kwargs)
        else:
            super().add_datasample(*args, **kwargs)

    @master_only
    def fomo_add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Optional[DetDataSample] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: int = 0,
        out_file: Optional[str] = None,
        pred_score_thr: float = 0.3,
        step: int = 0,
    ) -> None:
        self.pred_score_thr = pred_score_thr
        image = image.clip(0, 255)
        classes = self.dataset_meta.get('classes', None)
        plaettle = self.dataset_meta.get('palette', (0, 255, 0))

        if data_sample is not None:
            data_sample = data_sample.cpu()

        gt_img = None
        pred_img = None

        if draw_gt and data_sample is not None:
            gt_img = image
            if 'gt_instances' in data_sample:
                gt_img = self._draw_fomo_instances(gt_img, data_sample, classes=classes, plaettle=plaettle)

        if draw_pred and data_sample is not None:
            pred_img = image
            if 'pred_instances' in data_sample:
                pred_img = self._draw_fomo_instances(
                    pred_img, data_sample, bbox=False, classes=classes, plaettle=plaettle
                )
        if gt_img is not None and pred_img is not None:
            drawn_img = np.concatenate((gt_img, pred_img), axis=1)

        elif gt_img is not None:
            drawn_img = gt_img
        elif pred_img is not None:
            drawn_img = pred_img
        else:
            drawn_img = image

        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        if out_file:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)

    def _draw_fomo_instances(
        self,
        img: np.ndarray,
        data_sample: DetDataSample,
        bbox: bool = True,
        classes: Optional[Sequence[str]] = None,
        plaettle: Optional[Sequence[Tuple[int, ...]]] = None,
    ) -> np.ndarray:
        self.set_image(img)
        if bbox:
            instances: InstanceData = data_sample.gt_instances
        else:
            instances: InstanceData = data_sample.pred_instances
        ori_shape = data_sample.metainfo['ori_shape']
        img_shape = data_sample.metainfo['img_shape']
        if bbox and 'bboxes' in instances:
            bboxes: List[List[float]] = instances.bboxes
            labels: List[int] = instances.labels
            points = []
            texts = []
            for idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                x = (x1 + x2) / 2 / img_shape[1] * ori_shape[1]
                y = (y1 + y2) / 2 / img_shape[0] * ori_shape[0]
                points.append([x, y])
                texts.append(classes[labels[idx].item()])

            if len(points):
                self.draw_points(np.asarray(points), sizes=120)

                self.draw_texts(texts, np.asarray(points), font_sizes=30)

        elif 'pred' in instances:
            preds = instances.pred
            # labelss = instances.labels
            points = []
            for pred in preds:
                pred = pred.permute(0, 2, 3, 1).cpu().numpy()[0]
                H, W, C = pred.shape
                mask = pred[..., 1:] > self.pred_score_thr
                mask = np.any(mask, axis=2)
                mask = np.repeat(np.expand_dims(mask, -1), 3, axis=-1)
                pred = np.ma.array(pred, mask=~mask, keep_mask=True, copy=True, fill_value=0)

                pred_max = np.argmax(pred, axis=-1)

                pred_condition = np.where(pred_max > 0)
                pred_index = np.stack(pred_condition, axis=1)
                texts = []
                for i in pred_index:
                    idx = pred_max[i[0], i[1]]
                    texts.append(classes[idx - 1])
                if len(pred_index):
                    points = (pred_index + 0.5) / np.asarray([H, W]) * np.asarray(ori_shape)
                    self.draw_points(points, colors='r')
                    self.draw_texts(texts, points, font_sizes=30, colors='r')

        return self.get_image()


@VISUALIZERS.register_module()
class SensorClsVisualizer(Visualizer):
    """Universal Visualizer for classification task.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        fig_save_cfg (dict): Keyword parameters of figure for saving.
            Defaults to empty dict.
        fig_show_cfg (dict): Keyword parameters of figure for showing.
            Defaults to empty dict.

    Examples:
        >>> import torch
        >>> import mmcv
        >>> from pathlib import Path
        >>> from mmcls.visualization import ClsVisualizer
        >>> from mmcls.structures import ClsDataSample
        >>> # Example image
        >>> img = mmcv.imread("./demo/bird.JPEG", channel_order='rgb')
        >>> # Example annotation
        >>> data_sample = ClsDataSample().set_gt_label(1).set_pred_label(1).\
        ...     set_pred_score(torch.tensor([0.1, 0.8, 0.1]))
        >>> # Setup the visualizer
        >>> vis = ClsVisualizer(
        ...     save_dir="./outputs",
        ...     vis_backends=[dict(type='LocalVisBackend')])
        >>> # Set classes names
        >>> vis.dataset_meta = {'classes': ['cat', 'bird', 'dog']}
        >>> # Show the example image with annotation in a figure.
        >>> # And it will ignore all preset storage backends.
        >>> vis.add_datasample('res', img, data_sample, show=True)
        >>> # Save the visualization result by the specified storage backends.
        >>> vis.add_datasample('res', img, data_sample)
        >>> assert Path('./outputs/vis_data/vis_image/res_0.png').exists()
        >>> # Save another visualization result with the same name.
        >>> vis.add_datasample('res', img, data_sample, step=1)
        >>> assert Path('./outputs/vis_data/vis_image/res_1.png').exists()
    """

    @master_only
    def add_datasample(
        self,
        name: str,
        data: np.ndarray,
        data_sample: Optional[ClsDataSample] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        draw_score: bool = True,
        show: bool = False,
        text_cfg: dict = dict(),
        wait_time: float = 0,
        out_file: Optional[str] = None,
        step: int = 0,
    ) -> None:
        """Draw datasample and save to all backends.

        - If ``out_file`` is specified, all storage backends are ignored
          and save the image to the ``out_file``.
        - If ``show`` is True, plot the result image in a window, please
          confirm you are able to access the graphical interface.

        Args:
            name (str): The image identifier.
            data (np.ndarray): The data to draw.
            data_sample (:obj:`ClsDataSample`, optional): The annotation of the
                data. Defaults to None.
            draw_gt (bool): Whether to draw ground truth labels.
                Defaults to True.
            draw_pred (bool): Whether to draw prediction labels.
                Defaults to True.
            draw_score (bool): Whether to draw the prediction scores
                of prediction categories. Defaults to True.
            show (bool): Whether to display the drawn image. Defaults to False.
            text_cfg (dict): Extra text setting, which accepts
                arguments of :attr:`mmengine.Visualizer.draw_texts`.
                Defaults to an empty dict.
            wait_time (float): The interval of show (s). Defaults to 0, which
                means "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        classes = None
        if self.dataset_meta is not None:
            classes = self.dataset_meta.get('classes', None)

        sensors = data_sample.sensors
        uints = []
        uints = [sensor['units'] for sensor in sensors if sensor['units'] not in uints]

        # slice the data into different sensors
        inputs = [data[i :: len(sensors)] for i in range(len(sensors))]

        _, axs = plt.subplots(len(uints), 1)

        texts = []
        for j, input in enumerate(inputs):
            if len(uints) > 1:
                index = uints.index(sensors[j]['units'])
                ax = axs[index]
            else:
                ax = axs

            ax.plot(input, label=sensors[j]['name'])
            ax.set_ylabel(sensors[j]['units'])

        if draw_gt and 'gt_label' in data_sample:
            gt_label = data_sample.gt_label
            idx = gt_label.label.tolist()
            class_labels = [''] * len(idx)
            if classes is not None:
                class_labels = [f' ({classes[i]})' for i in idx]
            labels = [str(idx[i]) + class_labels[i] for i in range(len(idx))]
            prefix = 'Ground truth: '
            texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

        if draw_pred and 'pred_label' in data_sample:
            pred_label = data_sample.pred_label
            idx = pred_label.label.tolist()
            score_labels = [''] * len(idx)
            class_labels = [''] * len(idx)
            if draw_score and 'score' in pred_label:
                score_labels = [f', {pred_label.score[i].item():.2f}' for i in idx]

            if classes is not None:
                class_labels = [f' ({classes[i]})' for i in idx]

            labels = [str(idx[i]) + score_labels[i] + class_labels[i] for i in range(len(idx))]
            prefix = 'Prediction: '
            texts.append(prefix + ('\n' + ' ' * len(prefix)).join(labels))

        plt.set_title(texts)
        plt.tight_layout()
        plt.legend()
        fig = plt.gcf()
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
        self.set_image(image)
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)
