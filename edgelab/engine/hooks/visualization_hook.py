import os
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine
import mmengine.fileio as fileio
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmengine.hooks import Hook
from edgelab.registry import HOOKS
from mmdet.engine.hooks import DetVisualizationHook
from mmpose.structures import PoseDataSample, merge_data_samples


@HOOKS.register_module()
class Posevisualization(Hook):

    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        kpt_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.kpt_thr = kpt_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.enable = enable
        self.out_dir = out_dir
        self._test_index = 0
        self.backend_args = backend_args

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[PoseDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        if self.out_dir is not None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp,
                                        self.out_dir)
            mmengine.mkdir_or_exist(self.out_dir)

        self._visualizer.set_dataset_meta(runner.test_evaluator.dataset_meta)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.get('image_file')[0]
            img_path = img_path if isinstance(img_path, str) else img_path[0]
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            data_sample = merge_data_samples([data_sample])

            out_file = None
            if self.out_dir is not None:
                out_file_name, postfix = os.path.basename(img_path).rsplit(
                    '.', 1)
                index = len([
                    fname for fname in os.listdir(self.out_dir)
                    if fname.startswith(out_file_name)
                ])
                out_file = f'{out_file_name}_{index}.{postfix}'
                out_file = os.path.join(self.out_dir, out_file)
            from mmpose.visualization import PoseLocalVisualizer
            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                out_file=out_file,
                step=self._test_index)


@HOOKS.register_module()
class DetFomoVisualizationHook(DetVisualizationHook):

    def __init__(self, *args, fomo: bool = False, **kwarg):
        super().__init__(*args, **kwarg)
        self.fomo = fomo

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        if self.fomo:
            if self.draw is False:
                return

            # There is no guarantee that the same batch of images
            # is visualized for each evaluation.
            total_curr_iter = runner.iter + batch_idx

            # Visualize only the first data
            img_path = outputs[0].img_path
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            if total_curr_iter % self.interval == 0:
                self._visualizer.add_datasample(
                    osp.basename(img_path) if self.show else 'val_img',
                    img,
                    data_sample=outputs[0],
                    show=self.show,
                    wait_time=self.wait_time,
                    pred_score_thr=self.score_thr,
                    step=total_curr_iter)
        else:
            return super().after_val_iter(runner, batch_idx, data_batch, outputs)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
    
        if self.fomo:
            pass
        else:
            return super().after_test_iter(runner, batch_idx, data_batch, outputs)