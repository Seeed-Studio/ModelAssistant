# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure

from mmengine.dataset import BaseDataset
from mmengine.dist import master_only
from mmengine.visualization import Visualizer
from mmengine.visualization.utils import img_from_canvas
from mmengine.structures import InstanceData
from sscma.visualization import DetLocalVisualizer
from sscma.structures import DataSample, DetDataSample
from sscma.utils import (
    simplecv_imresize,
    simplecv_imrescale,
    simplecv_imread,
    simplecv_imwrite,
)


def get_adaptive_scale(
    img_shape: Tuple[int, int], min_scale: float = 0.3, max_scale: float = 3.0
) -> float:
    """Get adaptive scale according to image shape.

    The target scale depends on the the short edge length of the image. If the
    short edge length equals 224, the output is 1.0. And output linear scales
    according the short edge length.

    You can also specify the minimum scale and the maximum scale to limit the
    linear scale.

    Args:
        img_shape (Tuple[int, int]): The shape of the canvas image.
        min_size (int): The minimum scale. Defaults to 0.3.
        max_size (int): The maximum scale. Defaults to 3.0.

    Returns:
        int: The adaptive scale.
    """
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.0
    return min(max(scale, min_scale), max_scale)


def create_figure(*args, margin=False, **kwargs) -> "Figure":
    """Create a independent figure.

    Different from the :func:`plt.figure`, the figure from this function won't
    be managed by matplotlib. And it has
    :obj:`matplotlib.backends.backend_agg.FigureCanvasAgg`, and therefore, you
    can use the ``canvas`` attribute to get access the drawn image.

    Args:
        *args: All positional arguments of :class:`matplotlib.figure.Figure`.
        margin: Whether to reserve the white edges of the figure.
            Defaults to False.
        **kwargs: All keyword arguments of :class:`matplotlib.figure.Figure`.

    Return:
        matplotlib.figure.Figure: The created figure.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(*args, **kwargs)
    FigureCanvasAgg(figure)

    if not margin:
        # remove white edges by set subplot margin
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return figure


class UniversalVisualizer(Visualizer):
    """Universal Visualizer for multiple tasks.

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
    """

    DEFAULT_TEXT_CFG = {
        "family": "monospace",
        "color": "white",
        "bbox": dict(facecolor="black", alpha=0.5, boxstyle="Round"),
        "verticalalignment": "top",
        "horizontalalignment": "left",
    }

    @master_only
    def visualize_cls(
        self,
        image: np.ndarray,
        data_sample: DataSample,
        classes: Optional[Sequence[str]] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        draw_score: bool = True,
        resize: Optional[int] = None,
        rescale_factor: Optional[float] = None,
        text_cfg: dict = dict(),
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        name: str = "",
        step: int = 0,
    ) -> None:
        """Visualize image classification result.

        This method will draw an text box on the input image to visualize the
        information about image classification, like the ground-truth label and
        prediction label.

        Args:
            image (np.ndarray): The image to draw. The format should be RGB.
            data_sample (:obj:`DataSample`): The annotation of the image.
            classes (Sequence[str], optional): The categories names.
                Defaults to None.
            draw_gt (bool): Whether to draw ground-truth labels.
                Defaults to True.
            draw_pred (bool): Whether to draw prediction labels.
                Defaults to True.
            draw_score (bool): Whether to draw the prediction scores
                of prediction categories. Defaults to True.
            resize (int, optional): Resize the short edge of the image to the
                specified length before visualization. Defaults to None.
            rescale_factor (float, optional): Rescale the image by the rescale
                factor before visualization. Defaults to None.
            text_cfg (dict): Extra text setting, which accepts
                arguments of :meth:`mmengine.Visualizer.draw_texts`.
                Defaults to an empty dict.
            show (bool): Whether to display the drawn image in a window, please
                confirm your are able to access the graphical interface.
                Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            name (str): The image identifier. It's useful when using the
                storage backends of the visualizer to save or display the
                image. Defaults to an empty string.
            step (int): The global step value. It's useful to record a
                series of visualization results for the same image with the
                storage backends. Defaults to 0.

        Returns:
            np.ndarray: The visualization image.
        """
        if self.dataset_meta is not None:
            classes = classes or self.dataset_meta.get("classes", None)

        if resize is not None:
            h, w = image.shape[:2]
            if w < h:
                image = simplecv_imresize(image, (resize, resize * h // w))
            else:
                image = simplecv_imresize(image, (resize * w // h, resize))
        elif rescale_factor is not None:
            image = simplecv_imrescale(image, rescale_factor)

        texts = []
        self.set_image(image)

        if draw_gt and "gt_label" in data_sample:
            idx = data_sample.gt_label.tolist()
            class_labels = [""] * len(idx)
            if classes is not None:
                class_labels = [f" ({classes[i]})" for i in idx]
            labels = [str(idx[i]) + class_labels[i] for i in range(len(idx))]
            prefix = "Ground truth: "
            texts.append(prefix + ("\n" + " " * len(prefix)).join(labels))

        if draw_pred and "pred_label" in data_sample:
            idx = data_sample.pred_label.tolist()
            score_labels = [""] * len(idx)
            class_labels = [""] * len(idx)
            if draw_score and "pred_score" in data_sample:
                score_labels = [
                    f", {data_sample.pred_score[i].item():.2f}" for i in idx
                ]

            if classes is not None:
                class_labels = [f" ({classes[i]})" for i in idx]

            labels = [
                str(idx[i]) + score_labels[i] + class_labels[i] for i in range(len(idx))
            ]
            prefix = "Prediction: "
            texts.append(prefix + ("\n" + " " * len(prefix)).join(labels))

        img_scale = get_adaptive_scale(image.shape[:2])
        text_cfg = {
            "size": int(img_scale * 7),
            **self.DEFAULT_TEXT_CFG,
            **text_cfg,
        }
        self.ax_save.text(
            img_scale * 5,
            img_scale * 5,
            "\n".join(texts),
            **text_cfg,
        )
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            simplecv_imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img

    @master_only
    def visualize_image_retrieval(
        self,
        image: np.ndarray,
        data_sample: DataSample,
        prototype_dataset: BaseDataset,
        topk: int = 1,
        draw_score: bool = True,
        resize: Optional[int] = None,
        text_cfg: dict = dict(),
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        name: Optional[str] = "",
        step: int = 0,
    ) -> None:
        """Visualize image retrieval result.

        This method will draw the input image and the images retrieved from the
        prototype dataset.

        Args:
            image (np.ndarray): The image to draw. The format should be RGB.
            data_sample (:obj:`DataSample`): The annotation of the image.
            prototype_dataset (:obj:`BaseDataset`): The prototype dataset.
                It should have `get_data_info` method and return a dict
                includes `img_path`.
            draw_score (bool): Whether to draw the match scores of the
                retrieved images. Defaults to True.
            resize (int, optional): Resize the long edge of the image to the
                specified length before visualization. Defaults to None.
            text_cfg (dict): Extra text setting, which accepts arguments of
                :func:`plt.text`. Defaults to an empty dict.
            show (bool): Whether to display the drawn image in a window, please
                confirm your are able to access the graphical interface.
                Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            name (str): The image identifier. It's useful when using the
                storage backends of the visualizer to save or display the
                image. Defaults to an empty string.
            step (int): The global step value. It's useful to record a
                series of visualization results for the same image with the
                storage backends. Defaults to 0.

        Returns:
            np.ndarray: The visualization image.
        """
        text_cfg = {**self.DEFAULT_TEXT_CFG, **text_cfg}
        if resize is not None:
            image = simplecv_imrescale(image, (resize, resize))

        match_scores, indices = torch.topk(data_sample.pred_score, k=topk)

        figure = create_figure(margin=True)
        gs = figure.add_gridspec(2, topk)
        query_plot = figure.add_subplot(gs[0, :])
        query_plot.axis(False)
        query_plot.imshow(image)

        for k, (score, sample_idx) in enumerate(zip(match_scores, indices)):
            sample = prototype_dataset.get_data_info(sample_idx.item())
            value_image = simplecv_imread(sample["img_path"])[..., ::-1]
            value_plot = figure.add_subplot(gs[1, k])
            value_plot.axis(False)
            value_plot.imshow(value_image)
            if draw_score:
                value_plot.text(
                    5,
                    5,
                    f"{score:.2f}",
                    **text_cfg,
                )
        drawn_img = img_from_canvas(figure.canvas)
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            simplecv_imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img

    def add_mask_to_image(
        self,
        image: np.ndarray,
        data_sample: DataSample,
        resize: Union[int, Tuple[int]] = 224,
        color: Union[str, Tuple[int]] = "black",
        alpha: Union[int, float] = 0.8,
    ) -> np.ndarray:
        if isinstance(resize, int):
            resize = (resize, resize)

        image = simplecv_imresize(image, resize)
        self.set_image(image)

        if isinstance(data_sample.mask, np.ndarray):
            data_sample.mask = torch.tensor(data_sample.mask)
        mask = data_sample.mask.float()[None, None, ...]
        mask_ = F.interpolate(mask, image.shape[:2], mode="nearest")[0, 0]

        self.draw_binary_masks(mask_.bool(), colors=color, alphas=alpha)

        drawn_img = self.get_image()
        return drawn_img

    @master_only
    def visualize_masked_image(
        self,
        image: np.ndarray,
        data_sample: DataSample,
        resize: Union[int, Tuple[int]] = 224,
        color: Union[str, Tuple[int]] = "black",
        alpha: Union[int, float] = 0.8,
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        name: str = "",
        step: int = 0,
    ) -> None:
        """Visualize masked image.

        This method will draw an image with binary mask.

        Args:
            image (np.ndarray): The image to draw. The format should be RGB.
            data_sample (:obj:`DataSample`): The annotation of the image.
            resize (int | Tuple[int]): Resize the input image to the specified
                shape. Defaults to 224.
            color (str | Tuple[int]): The color of the binary mask.
                Defaults to "black".
            alpha (int | float): The transparency of the mask. Defaults to 0.8.
            show (bool): Whether to display the drawn image in a window, please
                confirm your are able to access the graphical interface.
                Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            name (str): The image identifier. It's useful when using the
                storage backends of the visualizer to save or display the
                image. Defaults to an empty string.
            step (int): The global step value. It's useful to record a
                series of visualization results for the same image with the
                storage backends. Defaults to 0.

        Returns:
            np.ndarray: The visualization image.
        """
        drawn_img = self.add_mask_to_image(
            image=image,
            data_sample=data_sample,
            resize=resize,
            color=color,
            alpha=alpha,
        )

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            simplecv_imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img

    @master_only
    def visualize_image_caption(
        self,
        image: np.ndarray,
        data_sample: DataSample,
        resize: Optional[int] = None,
        text_cfg: dict = dict(),
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        name: Optional[str] = "",
        step: int = 0,
    ) -> None:
        """Visualize image caption result.

        This method will draw the input image and the images caption.

        Args:
            image (np.ndarray): The image to draw. The format should be RGB.
            data_sample (:obj:`DataSample`): The annotation of the image.
            resize (int, optional): Resize the long edge of the image to the
                specified length before visualization. Defaults to None.
            text_cfg (dict): Extra text setting, which accepts arguments of
                :func:`plt.text`. Defaults to an empty dict.
            show (bool): Whether to display the drawn image in a window, please
                confirm your are able to access the graphical interface.
                Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            name (str): The image identifier. It's useful when using the
                storage backends of the visualizer to save or display the
                image. Defaults to an empty string.
            step (int): The global step value. It's useful to record a
                series of visualization results for the same image with the
                storage backends. Defaults to 0.

        Returns:
            np.ndarray: The visualization image.
        """
        text_cfg = {**self.DEFAULT_TEXT_CFG, **text_cfg}

        if resize is not None:
            h, w = image.shape[:2]
            if w < h:
                image = simplecv_imresize(image, (resize, resize * h // w))
            else:
                image = simplecv_imresize(image, (resize * w // h, resize))

        self.set_image(image)

        img_scale = get_adaptive_scale(image.shape[:2])
        text_cfg = {
            "size": int(img_scale * 7),
            **self.DEFAULT_TEXT_CFG,
            **text_cfg,
        }
        self.ax_save.text(
            img_scale * 5,
            img_scale * 5,
            data_sample.get("pred_caption"),
            wrap=True,
            **text_cfg,
        )
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            simplecv_imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img

    @master_only
    def visualize_vqa(
        self,
        image: np.ndarray,
        data_sample: DataSample,
        resize: Optional[int] = None,
        text_cfg: dict = dict(),
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        name: Optional[str] = "",
        step: int = 0,
    ) -> None:
        """Visualize visual question answering result.

        This method will draw the input image, question and answer.

        Args:
            image (np.ndarray): The image to draw. The format should be RGB.
            data_sample (:obj:`DataSample`): The annotation of the image.
            resize (int, optional): Resize the long edge of the image to the
                specified length before visualization. Defaults to None.
            text_cfg (dict): Extra text setting, which accepts arguments of
                :func:`plt.text`. Defaults to an empty dict.
            show (bool): Whether to display the drawn image in a window, please
                confirm your are able to access the graphical interface.
                Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            name (str): The image identifier. It's useful when using the
                storage backends of the visualizer to save or display the
                image. Defaults to an empty string.
            step (int): The global step value. It's useful to record a
                series of visualization results for the same image with the
                storage backends. Defaults to 0.

        Returns:
            np.ndarray: The visualization image.
        """
        text_cfg = {**self.DEFAULT_TEXT_CFG, **text_cfg}

        if resize is not None:
            h, w = image.shape[:2]
            if w < h:
                image = simplecv_imresize(image, (resize, resize * h // w))
            else:
                image = simplecv_imresize(image, (resize * w // h, resize))

        self.set_image(image)

        img_scale = get_adaptive_scale(image.shape[:2])
        text_cfg = {
            "size": int(img_scale * 7),
            **self.DEFAULT_TEXT_CFG,
            **text_cfg,
        }
        text = (
            f'Q: {data_sample.get("question")}\n' f'A: {data_sample.get("pred_answer")}'
        )
        self.ax_save.text(
            img_scale * 5,
            img_scale * 5,
            text,
            wrap=True,
            **text_cfg,
        )
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            simplecv_imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img

    @master_only
    def visualize_visual_grounding(
        self,
        image: np.ndarray,
        data_sample: DataSample,
        resize: Optional[int] = None,
        text_cfg: dict = dict(),
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        name: Optional[str] = "",
        line_width: Union[int, float] = 3,
        bbox_color: Union[str, tuple] = "green",
        step: int = 0,
    ) -> None:
        """Visualize visual grounding result.

        This method will draw the input image, bbox and the object.

        Args:
            image (np.ndarray): The image to draw. The format should be RGB.
            data_sample (:obj:`DataSample`): The annotation of the image.
            resize (int, optional): Resize the long edge of the image to the
                specified length before visualization. Defaults to None.
            text_cfg (dict): Extra text setting, which accepts arguments of
                :func:`plt.text`. Defaults to an empty dict.
            show (bool): Whether to display the drawn image in a window, please
                confirm your are able to access the graphical interface.
                Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            name (str): The image identifier. It's useful when using the
                storage backends of the visualizer to save or display the
                image. Defaults to an empty string.
            step (int): The global step value. It's useful to record a
                series of visualization results for the same image with the
                storage backends. Defaults to 0.

        Returns:
            np.ndarray: The visualization image.
        """
        text_cfg = {**self.DEFAULT_TEXT_CFG, **text_cfg}

        gt_bboxes = data_sample.get("gt_bboxes")
        pred_bboxes = data_sample.get("pred_bboxes")
        if resize is not None:
            h, w = image.shape[:2]
            if w < h:
                image, w_scale, h_scale = simplecv_imresize(
                    image, (resize, resize * h // w), return_scale=True
                )
            else:
                image, w_scale, h_scale = simplecv_imresize(
                    image, (resize * w // h, resize), return_scale=True
                )
            pred_bboxes[:, ::2] *= w_scale
            pred_bboxes[:, 1::2] *= h_scale
            if gt_bboxes is not None:
                gt_bboxes[:, ::2] *= w_scale
                gt_bboxes[:, 1::2] *= h_scale

        self.set_image(image)
        # Avoid the line-width limit in the base classes.
        self._default_font_size = 1e3
        self.draw_bboxes(pred_bboxes, line_widths=line_width, edge_colors=bbox_color)
        if gt_bboxes is not None:
            self.draw_bboxes(gt_bboxes, line_widths=line_width, edge_colors="blue")

        img_scale = get_adaptive_scale(image.shape[:2])
        text_cfg = {
            "size": int(img_scale * 7),
            **self.DEFAULT_TEXT_CFG,
            **text_cfg,
        }

        text_positions = pred_bboxes[:, :2] + line_width
        for i in range(pred_bboxes.size(0)):
            self.ax_save.text(
                text_positions[i, 0] + line_width,
                text_positions[i, 1] + line_width,
                data_sample.get("text"),
                **text_cfg,
            )
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            simplecv_imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img

    @master_only
    def visualize_t2i_retrieval(
        self,
        text: str,
        data_sample: DataSample,
        prototype_dataset: BaseDataset,
        topk: int = 1,
        draw_score: bool = True,
        text_cfg: dict = dict(),
        fig_cfg: dict = dict(),
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        name: Optional[str] = "",
        step: int = 0,
    ) -> None:
        """Visualize Text-To-Image retrieval result.

        This method will draw the input text and the images retrieved from the
        prototype dataset.

        Args:
            image (np.ndarray): The image to draw. The format should be RGB.
            data_sample (:obj:`DataSample`): The annotation of the image.
            prototype_dataset (:obj:`BaseDataset`): The prototype dataset.
                It should have `get_data_info` method and return a dict
                includes `img_path`.
            topk (int): To visualize the topk matching items. Defaults to 1.
            draw_score (bool): Whether to draw the match scores of the
                retrieved images. Defaults to True.
            text_cfg (dict): Extra text setting, which accepts arguments of
                :func:`plt.text`. Defaults to an empty dict.
            fig_cfg (dict): Extra figure setting, which accepts arguments of
                :func:`plt.Figure`. Defaults to an empty dict.
            show (bool): Whether to display the drawn image in a window, please
                confirm your are able to access the graphical interface.
                Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            name (str): The image identifier. It's useful when using the
                storage backends of the visualizer to save or display the
                image. Defaults to an empty string.
            step (int): The global step value. It's useful to record a
                series of visualization results for the same image with the
                storage backends. Defaults to 0.

        Returns:
            np.ndarray: The visualization image.
        """
        text_cfg = {**self.DEFAULT_TEXT_CFG, **text_cfg}

        match_scores, indices = torch.topk(data_sample.pred_score, k=topk)

        figure = create_figure(margin=True, **fig_cfg)
        figure.suptitle(text)
        gs = figure.add_gridspec(1, topk)

        for k, (score, sample_idx) in enumerate(zip(match_scores, indices)):
            sample = prototype_dataset.get_data_info(sample_idx.item())
            value_image = simplecv_imread(sample["img_path"])[..., ::-1]
            value_plot = figure.add_subplot(gs[0, k])
            value_plot.axis(False)
            value_plot.imshow(value_image)
            if draw_score:
                value_plot.text(
                    5,
                    5,
                    f"{score:.2f}",
                    **text_cfg,
                )
        drawn_img = img_from_canvas(figure.canvas)
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            simplecv_imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img

    @master_only
    def visualize_i2t_retrieval(
        self,
        image: np.ndarray,
        data_sample: DataSample,
        prototype_dataset: Sequence[str],
        topk: int = 1,
        draw_score: bool = True,
        resize: Optional[int] = None,
        text_cfg: dict = dict(),
        show: bool = False,
        wait_time: float = 0,
        out_file: Optional[str] = None,
        name: str = "",
        step: int = 0,
    ) -> None:
        """Visualize Image-To-Text retrieval result.

        This method will draw the input image and the texts retrieved from the
        prototype dataset.

        Args:
            image (np.ndarray): The image to draw. The format should be RGB.
            data_sample (:obj:`DataSample`): The annotation of the image.
            prototype_dataset (Sequence[str]): The prototype dataset.
                It should be a list of texts.
            topk (int): To visualize the topk matching items. Defaults to 1.
            draw_score (bool): Whether to draw the prediction scores
                of prediction categories. Defaults to True.
            resize (int, optional): Resize the short edge of the image to the
                specified length before visualization. Defaults to None.
            text_cfg (dict): Extra text setting, which accepts
                arguments of :meth:`mmengine.Visualizer.draw_texts`.
                Defaults to an empty dict.
            show (bool): Whether to display the drawn image in a window, please
                confirm your are able to access the graphical interface.
                Defaults to False.
            wait_time (float): The display time (s). Defaults to 0, which means
                "forever".
            out_file (str, optional): Extra path to save the visualization
                result. If specified, the visualizer will only save the result
                image to the out_file and ignore its storage backends.
                Defaults to None.
            name (str): The image identifier. It's useful when using the
                storage backends of the visualizer to save or display the
                image. Defaults to an empty string.
            step (int): The global step value. It's useful to record a
                series of visualization results for the same image with the
                storage backends. Defaults to 0.

        Returns:
            np.ndarray: The visualization image.
        """
        if resize is not None:
            h, w = image.shape[:2]
            if w < h:
                image = simplecv_imresize(image, (resize, resize * h // w))
            else:
                image = simplecv_imresize(image, (resize * w // h, resize))

        self.set_image(image)

        match_scores, indices = torch.topk(data_sample.pred_score, k=topk)
        texts = []
        for score, sample_idx in zip(match_scores, indices):
            text = prototype_dataset[sample_idx.item()]
            if draw_score:
                text = f"{score:.2f} " + text
            texts.append(text)

        img_scale = get_adaptive_scale(image.shape[:2])
        text_cfg = {
            "size": int(img_scale * 7),
            **self.DEFAULT_TEXT_CFG,
            **text_cfg,
        }
        self.ax_save.text(
            img_scale * 5,
            img_scale * 5,
            "\n".join(texts),
            **text_cfg,
        )
        drawn_img = self.get_image()

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            # save the image to the target file instead of vis_backends
            simplecv_imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step=step)

        return drawn_img


class FomoLocalVisualizer(DetLocalVisualizer):
    """Unified Fomo and target detection visualization classes."""

    def __init__(self, name="v", *args, fomo=False, **kwargs) -> None:
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
        classes = self.dataset_meta.get("classes", None)
        plaettle = self.dataset_meta.get("palette", (0, 255, 0))

        if data_sample is not None:
            data_sample = data_sample.cpu()

        gt_img = None
        pred_img = None

        if draw_gt and data_sample is not None:
            gt_img = image
            if "gt_instances" in data_sample:
                gt_img = self._draw_fomo_instances(
                    gt_img, data_sample, classes=classes, plaettle=plaettle
                )

        if draw_pred and data_sample is not None:
            pred_img = image
            if "pred_instances" in data_sample:
                pred_img = self._draw_fomo_instances(
                    pred_img,
                    data_sample,
                    bbox=False,
                    classes=classes,
                    plaettle=plaettle,
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
            cv2.imwrite(drawn_img[..., ::-1], out_file)
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
        ori_shape = data_sample.metainfo["ori_shape"]
        img_shape = data_sample.metainfo["img_shape"]
        if bbox and "bboxes" in instances:
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

        elif "pred" in instances:
            preds = instances.pred
            # labelss = instances.labels
            points = []
            for pred in preds:
                pred = pred.permute(0, 2, 3, 1).cpu().numpy()[0]
                H, W, C = pred.shape
                mask = pred[..., 1:] > self.pred_score_thr
                mask = np.any(mask, axis=2)
                mask = np.repeat(np.expand_dims(mask, -1), 3, axis=-1)
                pred = np.ma.array(
                    pred, mask=~mask, keep_mask=True, copy=True, fill_value=0
                )

                pred_max = np.argmax(pred, axis=-1)

                pred_condition = np.where(pred_max > 0)
                pred_index = np.stack(pred_condition, axis=1)
                texts = []
                for i in pred_index:
                    idx = pred_max[i[0], i[1]]
                    texts.append(classes[idx - 1])
                if len(pred_index):
                    points = (
                        (pred_index + 0.5) / np.asarray([H, W]) * np.asarray(ori_shape)
                    )
                    self.draw_points(points, colors="r")
                    self.draw_texts(texts, points, font_sizes=30, colors="r")

        return self.get_image()
