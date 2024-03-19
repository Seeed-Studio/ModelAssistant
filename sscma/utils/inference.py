import datetime
import os
import os.path as osp
import time
from typing import AnyStr, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnx
import torch
from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine import dump as resultdump
from mmengine.config import Config
from mmengine.evaluator import Evaluator
from mmengine.registry import MODELS
from mmengine.runner import Runner
from mmengine.structures import InstanceData
from mmengine.visualization.visualizer import Visualizer
from torch.utils.data import DataLoader
from tqdm.std import tqdm

from sscma.utils.cv import NMS, NMS_FREE, load_image

from .iot_camera import IoTCamera


def _get_adaptive_scale(img_shape: Tuple[int, int], min_scale: float = 0.1, max_scale: float = 10.0) -> float:
    """Get adaptive scale according to image shape.

    The target scale depends on the the short edge length of the image. If the
    short edge length equals 224, the output is 1.0. And output linear scales
    according the short edge length.

    You can also specify the minimum scale and the maximum scale to limit the
    linear scale.

    Args:
        img_shape (Tuple[int, int]): The shape of the canvas image.
        min_size (int): The minimum scale. Defaults to 0.1.
        max_size (int): The maximum scale. Defaults to 10.0.

    Returns:
        int: The adaptive scale.
    """
    short_edge_length = min(img_shape)
    scale = short_edge_length / 224.0
    return min(max(scale, min_scale), max_scale)


class Inter:
    def __init__(self, model: List or AnyStr or Tuple):
        if isinstance(model, list) or '.param' in model or '.bin' in model:
            try:
                import ncnn
            except ImportError:
                raise ImportError(
                    'You have not installed ncnn yet, please execute the "pip install ncnn" command to install and run again'
                )
            net = ncnn.Net()
            if isinstance(model, str):
                if '.bin' in model:
                    model = [model, model.replace('.bin', '.param')]
                else:
                    model = [model, model.replace('.param', '.bin')]
            for p in model:
                if p.endswith('param'):
                    param = p
                if p.endswith('bin'):
                    bin = p
            # load model weights
            net.load_param(param)
            net.load_model(bin)
            net.opt.use_vulkan_compute = False
            self.engine = 'ncnn'
            self._input_shape = [3, 640, 640]
        elif model.endswith('onnx'):
            try:
                import onnxruntime
            except ImportError:
                raise ImportError(
                    'You have not installed onnxruntime yet, please execute the "pip install onnxruntime" command to install and run again'
                )
            try:
                net = onnx.load(model)
                onnx.checker.check_model(net)
            except Exception:
                raise ValueError('onnx file have error,please check your onnx export code!')
            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if torch.cuda.is_available()
                else ['CPUExecutionProvider']
            )
            net = onnxruntime.InferenceSession(model, providers=providers)

            self._input_shape = net.get_inputs()[0].shape[1:]
            channels = self._input_shape.pop(0)
            self._input_type = net.get_inputs()[0].type
            self._input_shape.append(channels)
            self.engine = 'onnx'
        elif model.endswith('tflite'):
            try:
                import tensorflow as tf
            except ImportError:
                raise ImportError(
                    'You have not installed tensorflow yet, please execute the "pip install tensorflow" command to install and run again'
                )
            inter = tf.lite.Interpreter
            net = inter(model)
            self._input_shape = tuple(net.get_input_details()[0]['shape'][1:])
            net.allocate_tensors()
            self.engine = 'tf'
        else:
            raise 'model file input error'
        self.inter = net

    @property
    def input_shape(self):
        return self._input_shape

    def __call__(
        self,
        data: Union[np.array, torch.Tensor],
        input_name: AnyStr = 'input',
        output_name: AnyStr = 'output',
        result_num=1,
    ):
        if len(data.shape) == 3:
            C, H, W = data.shape
            if C not in [1, 3]:
                data = data.transpose(2, 0, 1)
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            data = np.array([data])  # add batch dim.
        elif len(data.shape) == 4:
            B, C, H, W = data.shape
            if C not in [1, 3]:
                data = data.transpose(0, 3, 1, 2)
            if isinstance(data, torch.Tensor):
                data = data.numpy()
        else:
            if isinstance(data, torch.Tensor):
                data = data.numpy()

        results = []
        if self.engine == 'onnx':  # onnx
            result = self.inter.run([self.inter.get_outputs()[0].name], {self.inter.get_inputs()[0].name: data})[0]
            results.append(result)
        elif self.engine == 'ncnn':  # ncnn
            import ncnn

            self.inter.opt.use_vulkan_compute = False
            extra = self.inter.create_extractor()
            input_name = self.inter.input_names()[0]
            output_name = self.inter.output_names()[0]
            extra.input(input_name, ncnn.Mat(data[0]))  # noqa
            result = extra.extract(output_name)[1]
            return [np.expand_dims(np.array(result), axis=0)]
        else:  # tf
            input_, outputs = self.inter.get_input_details()[0], [
                self.inter.get_output_details()[i] for i in range(result_num)
            ]
            input_int8 = input_['dtype'] == np.int8 or input_['dtype'] == np.uint8
            output_int8 = outputs[0]['dtype'] == np.int8 or outputs[0]['dtype'] == np.uint8
            data = data.transpose(0, 2, 3, 1) if len(data.shape) == 4 else data
            if input_int8:
                scale, zero_point = input_['quantization']
                data = (data / scale + zero_point).astype(np.int8)
            self.inter.set_tensor(input_['index'], data)
            self.inter.invoke()
            for output in outputs:
                result = self.inter.get_tensor(output['index'])
                if output_int8:
                    scale, zero_point = output['quantization']
                    result = (result.astype(np.float32) - zero_point) * scale
                results.append(result)

        return results


IMG_SUFFIX = ('.jpg', '.png', '.PNG', '.jpeg')
VIDEO_SUFFIX = ('.avi', '.mp4', '.mkv', '.flv', '.wmv', '.3gp')
IOT_DEVICE = ('sensorcap',)


class DataStream:
    def __init__(self, source: Union[int, str], shape: Optional[int or Tuple[int, int]] = None) -> None:
        if shape:
            self.gray = True if shape[-1] == 1 else False
            self.shape = shape[:-1]
        else:
            self.gray = False
            self.shape = shape
        self.file = None
        self.l = 0

        if isinstance(source, str):
            if osp.isdir(source):
                self.file = [osp.join(source, f) for f in os.listdir(source) if f.lower().endswith(IMG_SUFFIX)]
                self.l = len(self.file)
                self.file = iter(self.file)

            elif osp.isfile(source):
                if any([source.lower().endswith(mat) for mat in IMG_SUFFIX]):
                    self.file = [source]
                    self.l = len(self.file)
                    self.file = iter(self.file)
                elif any([source.lower().endswith(mat) for mat in VIDEO_SUFFIX]):
                    self.cap = cv2.VideoCapture(source)
            elif source.isdigit():
                self.cap = cv2.VideoCapture(int(source))
            elif source in IOT_DEVICE:
                self.cap = IoTCamera()
            else:
                raise
        elif isinstance(source, int):
            self.cap = cv2.VideoCapture(source)
        else:
            raise

    def __len__(self):
        return self.l if self.file else None

    def __iter__(self):
        return self

    def __next__(self):
        if self.file:
            f = next(self.file)
            img = load_image(f, shape=self.shape, mode='GRAY' if self.gray else 'RGB', normalized=True)

        else:
            while True:
                ret, img = self.cap.read()
                if ret:
                    break
                else:
                    time.sleep(0.1)

            if self.gray:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.expand_dims(img, axis=-1)

            if self.shape:
                img = cv2.resize(img, self.shape[::-1])

            img = (img / 255).astype(np.float32)

        return img


def build_target(pred_shape, ori_shape, gt_bboxs):
    """
    The target feature map constructed according to the size
    of the feature map output by the model
    bbox: xyxy
    """
    H, W, C = pred_shape

    target_data = torch.zeros(size=(1, *pred_shape))
    target_data[..., 0] = 1
    for b, bboxs in enumerate(gt_bboxs):
        for idx, bbox in enumerate(bboxs.bboxes):
            w = (bbox[2] + bbox[0]) / 2 / ori_shape[1]
            h = (bbox[3] + bbox[1]) / 2 / ori_shape[0]
            h, w = int(h.item() * H), int(w.item() * W)
            target_data[0, h, w, 0] = 0  # background
            target_data[0, h, w, bboxs.labels[idx] + 1] = 1  # label
    return target_data


class Infernce:
    """Model Reasoning Test Reasonable onnx, tflite, ncnn and other models."""

    def __init__(
        self,
        model: List or AnyStr or Tuple,
        dataloader: Union[DataLoader, str, int, None] = None,
        cfg: Optional[Config] = None,
        runner: Runner = None,
        dump: Optional[str] = None,
        source: Optional[str] = None,
        task: str = 'det',
        show: bool = False,
        save_dir: Optional[str] = None,
        audio: bool = False,
    ) -> None:
        # check source data
        assert not (source is None and dataloader is None), 'Both source and dataload cannot be None'

        self.class_name = (
            dataloader.dataset._metainfo['classes']
            if getattr(dataloader.dataset, '_metainfo', None) is not None
            else None
        )
        # load model
        self.model = Inter(model)
        # make dataloader
        self.source = source
        self.runner = runner
        self.dump = dump

        if source:
            self.dataloader = DataStream(source, shape=self.model.input_shape)
        else:
            self.dataloader = dataloader

        self.cfg = cfg
        if 'fomo' in self.cfg.visualizer:
            self.fomo = self.cfg.visualizer.fomo
        else:
            self.fomo = False

        self.show = show
        self.task = task
        self.save_dir = save_dir
        self.input_shape = self.model.input_shape
        self.init(cfg)

    def init(self, cfg):
        self.evaluator: Evaluator = self.runner.build_evaluator(self.cfg.get('val_evaluator'))
        if hasattr(self.dataloader, 'dataset'):
            self.evaluator.dataset_meta = self.dataloader.dataset.METAINFO
        if hasattr(cfg.model, 'data_preprocessor'):
            self.data_preprocess = MODELS.build(cfg.model.data_preprocessor)
        if hasattr(cfg, 'visualizer'):
            self.visualizer: Visualizer = Visualizer.get_current_instance()

    def post_process(self):
        pass

    def test(self) -> None:
        self.time_cost = 0
        self.preds = []
        img = None
        P = []
        R = []
        F1 = []

        for data in tqdm(self.dataloader):
            if not self.source:
                if hasattr(self, 'data_preprocess'):
                    data = self.data_preprocess(data, False)
                inputs = data['inputs'][0]
                if self.cfg.input_type == 'image':
                    data_samples = data['data_samples']
                    img_path = (
                        data_samples.get('img_path', None)
                        if isinstance(data_samples, dict)
                        else data_samples[0].get('img_path', None)
                    )
                    img = data['inputs'][0].permute(1, 2, 0).cpu().numpy()
            else:
                img = data
                inputs = data
                img_path = None

            t0 = time.time()

            preds = self.model(inputs)
            self.time_cost += time.time() - t0

            result = InstanceData()
            if self.task == 'pose':
                if self.source:
                    if img.dtype == np.float32:
                        img = img * 255
                    show_point(preds[0], img=img, show=self.show, save_path=self.save_dir)
                else:
                    show_point(
                        preds[0],
                        img_file=data['data_samples']['image_file'][0],
                        show=self.show,
                        save_path=self.save_dir,
                    )
                data['data_samples']['results'] = preds[0]
                self.evaluator.process(data_batch=data, data_samples=[data['data_samples']])
            elif self.task == 'det':
                img_path = data['data_samples'][0].get('img_path', None)
                if len(preds[0].shape) > 3:
                    preds = preds[0]
                elif len(preds[0].shape) > 2:
                    preds = preds[0][0]
                elif len(preds[0].shape) == 2:
                    preds = preds[0]
                else:
                    Warning('!!!')
                if self.fomo:
                    pred = preds[0]
                    H, W, C = pred.shape
                    mask = pred[..., 1:] > 0.7
                    mask = np.any(mask, axis=2)
                    mask = np.repeat(np.expand_dims(mask, -1), C, axis=-1)
                    pred = np.ma.array(pred, mask=~mask, keep_mask=True, copy=True, fill_value=0)

                    pred_max = np.argmax(pred, axis=-1)

                    pred_condition = np.where(pred_max > 0)
                    pred_index = np.stack(pred_condition, axis=1)
                    texts = []
                    for i in pred_index:
                        idx = pred_max[i[0], i[1]]
                        texts.append(idx - 1)
                    if len(pred_index):
                        points = (pred_index + 0.5) / np.asarray([H, W]) * np.asarray(self.input_shape[:-1])
                        show_point(points, img=img, labels=texts, show=self.show, img_file=img_path)
                    if not self.source:
                        ori_shape = data['data_samples'][0].ori_shape
                        # bboxes = data['data_samples'][0].gt_instances
                        # target = build_target(preds.shape[1:], (H*8, W*8), bboxes)
                        target = torch.as_tensor(data['data_samples'][0].fomo_mask[0])

                        data['data_samples'][0].pred_instances = InstanceData(
                            pred=tuple([torch.from_numpy(preds).permute(0, 3, 1, 2)]), labels=tuple([target])
                        )

                        self.evaluator.process(data_batch=data, data_samples=data['data_samples'])

                else:
                    # performs nms
                    if preds.shape[1] - len(self.class_name) == 4:
                        bbox, classes = preds[:, :4], preds[:, 4:]
                        preds = NMS_FREE(bbox, classes, 3000, conf_thres=20, bbox_format='xyxy')
                    else:
                        bbox, conf, classes = preds[:, :4], preds[:, 4], preds[:, 5:]
                        preds = NMS(bbox, conf, classes, conf_thres=20, bbox_format='xywh')

                    # self.visualizer.add_datasample("test_img", img, data['data_sample'], show=self.show)
                    # show det result and save result
                    show_det(
                        preds,
                        img=img,
                        img_file=img_path,
                        class_name=self.class_name,
                        shape=self.input_shape[:-1],
                        show=self.show,
                        save_path=self.save_dir,
                    )

                if not self.source and not self.fomo:
                    ori_shape = data['data_samples'][0].ori_shape
                    tmp = preds[:, :4]
                    tmp[:, 0::2] = tmp[:, 0::2] / self.input_shape[1] * ori_shape[1]
                    tmp[:, 1::2] = tmp[:, 1::2] / self.input_shape[0] * ori_shape[0]
                    result.bboxes = tmp
                    result.scores = preds[:, 4]
                    result.labels = preds[:, 5].type(torch.int)
                    # result.img_id = str(data['data_samples'][0].img_id)

                    for data_sample, pred_instances in zip(data['data_samples'], [result]):
                        data_sample.pred_instances = pred_instances
                    samplelist_boxtype2tensor(data)

                    self.evaluator.process(data_batch=data, data_samples=data['data_samples'])

            elif self.task == 'cls':
                img_path = data['data_samples'][0].get('img_path', None)
                label = np.argmax(preds[0], axis=1)
                data['data_samples'][0].set_pred_score(preds[0][0]).set_pred_label(label)
                self.evaluator.process(data_samples=data['data_samples'], data_batch=data)

                if self.cfg.input_type == 'sensor':
                    self.visualizer.add_datasample(name=0, data=inputs, data_sample=data['data_samples'][0])
                else:
                    if img.dtype == np.float32:
                        img = img * 255
                    img_scale = 1 / _get_adaptive_scale(img.shape[:2])
                    img = cv2.resize(img, (int(img.shape[1] * img_scale), int(img.shape[0] * img_scale)))
                    self.visualizer.set_image(img)
                    texts = self.visualizer.dataset_meta['classes'][label[0]] + ':' + str(preds[0][0][label[0]])

                    text_cfg = dict()
                    text_cfg = {
                        'texts': texts,
                        'positions': np.array([[0, 5]]),
                        'font_sizes': int(img.shape[0] / 20),
                        'font_families': 'monospace',
                        'colors': 'white',
                        'bboxes': dict(facecolor='black', alpha=0.5, boxstyle='Round'),
                        **text_cfg,
                    }
                    self.visualizer.draw_texts(**text_cfg)

                if self.show:
                    self.visualizer.show(backend='cv2')
            else:
                raise ValueError
        if not self.source:
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            if self.dump is not None and metrics is not None:
                resultdump(metrics, self.dump)
                print(metrics)
        if len(P):
            print('P:', sum(P) / len(P))
            print('R:', sum(R) / len(R))
            print('F1:', sum(F1) / len(F1))

        print(f'FPS: {len(self.dataloader)/self.time_cost:4f} fram/s')


def show_point(
    keypoints: Union[np.ndarray, Sequence[Sequence[int]], None] = None,
    img: Optional[np.ndarray] = None,
    img_file: Optional[str] = None,
    shape: Optional[Sequence[int]] = None,
    labels: Sequence[str] = None,
    win_name: str = 'test',
    save_path: bool = False,
    show: bool = False,
):
    # load image
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = load_image(img_file, shape=shape, mode='BGR').copy()

    H, W, _ = img.shape
    for idx, point in enumerate(keypoints):
        img = cv2.circle(img, (int(point[0] * W), int(point[1] * H)), 5, (255, 0, 0), -1)
        if labels:
            cv2.putText(
                img,
                str(labels[idx]),
                (int(point[0] * W), int(point[1] * H)),
                1,
                color=(0, 0, 255),
                thickness=1,
                fontScale=1,
            )
    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(500)

    if save_path:
        if img_file is None:
            img_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.jpg'
        else:
            img_name = osp.basename(img_file)
        cv2.imwrite(osp.join(save_path, img_name), img)


def show_det(
    pred: np.ndarray,
    img: Optional[np.ndarray] = None,
    img_file: Optional[str] = None,
    win_name='Detection',
    class_name=None,
    shape=None,
    save_path=False,
    show=False,
) -> np.ndarray:
    assert not (img is None and img_file is None), 'The img and img_file parameters cannot both be None'

    # load image
    if isinstance(img, np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = load_image(img_file, shape=shape, mode='BGR').copy()

    # plot the result
    for i in pred:
        x1, y1, x2, y2 = map(int, i[:4])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(
            img,
            class_name[int(i[5])] if class_name else 'None',
            (x1, y1),
            1,
            color=(0, 0, 255),
            thickness=1,
            fontScale=1,
        )
        cv2.putText(img, str(round(i[4].item(), 2)), (x1, y1 - 15), 1, color=(0, 0, 255), thickness=1, fontScale=1)
    if show:
        cv2.imshow(win_name, img)
        cv2.waitKey(500)

    if save_path:
        if img_file is None:
            img_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.jpg'
        else:
            img_name = osp.basename(img_file)
        cv2.imwrite(osp.join(save_path, img_name), img)

    return pred


if __name__ == '__main__':
    data = DataStream(0)
    data = iter(data)
    for img in data:
        cv2.imshow('aaa', img)
        cv2.waitKey(0)
