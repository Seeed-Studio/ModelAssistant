from typing import Dict, List, Optional, Sequence, Tuple
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.visualization import DetLocalVisualizer
import numpy as np
import mmcv
from edgelab.registry import VISUALIZERS


@VISUALIZERS.register_module()
class FomoLocalVisualizer(DetLocalVisualizer):
    """
    Unified Fomo and target detection visualization classes
    
    """

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
                gt_img = self._draw_fomo_instances(gt_img,
                                                   data_sample,
                                                   classes=classes,
                                                   plaettle=plaettle)

        if draw_pred and data_sample is not None:
            pred_img = image
            if 'pred_instances' in data_sample:
                pred_img = self._draw_fomo_instances(pred_img,
                                                     data_sample,
                                                     bbox=False,
                                                     classes=classes,
                                                     plaettle=plaettle)
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
            plaettle: Optional[Sequence[Tuple[int,
                                              ...]]] = None) -> np.ndarray:

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
            labelss = instances.labels
            points = []
            for pred in preds:
                pred = pred.permute(0, 2, 3, 1).cpu().numpy()[0]
                H, W, C = pred.shape
                mask = pred[..., 1:] > self.pred_score_thr
                mask = np.any(mask, axis=2)
                mask = np.repeat(np.expand_dims(mask, -1), 3, axis=-1)
                pred = np.ma.array(pred,
                                   mask=~mask,
                                   keep_mask=True,
                                   copy=True,
                                   fill_value=0)

                pred_max = np.argmax(pred, axis=-1)

                pred_condition = np.where(pred_max > 0)
                pred_index = np.stack(pred_condition, axis=1)
                texts = []
                for i in pred_index:
                    idx = pred_max[i[0], i[1]]
                    texts.append(classes[idx - 1])
                if len(pred_index):
                    points = (pred_index + 0.5) / np.asarray(
                        [H, W]) * np.asarray(ori_shape)
                    self.draw_points(points, colors='r')
                    self.draw_texts(texts, points, font_sizes=30, colors='r')

        return self.get_image()