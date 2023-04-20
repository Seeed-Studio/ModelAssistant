import warnings
from typing import Callable, List, Optional

import cv2
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.evaluation import EvalHook
from torch.utils.data import DataLoader

from edgelab.core.apis.mmdet.test import single_gpu_test_fomo


def show_result(result, img_path, classes):
    img = cv2.imread(img_path)
    H, W = img.shape[:-1]
    for i in result:
        w, h, label = i
        label = classes[label - 1]
        cv2.circle(img, (int(W * w), int(H * h)), 5, (0, 0, 255), 1)
        cv2.putText(img,
                    str(label),
                    org=(int(W * w), int(H * h)),
                    color=(255, 0, 0),
                    fontScale=1,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    cv2.imshow('img', img)
    cv2.waitKey(0)


@HOOKS.register_module()
class Feval(EvalHook):

    def __init__(self,
                 dataloader: DataLoader,
                 start: Optional[int] = None,
                 interval: int = 1,
                 by_epoch: bool = True,
                 fomo: bool = False,
                 save_best: Optional[str] = None,
                 rule: Optional[str] = None,
                 test_fn: Optional[Callable] = None,
                 greater_keys: Optional[List[str]] = None,
                 less_keys: Optional[List[str]] = None,
                 out_dir: Optional[str] = None,
                 file_client_args: Optional[dict] = None,
                 **eval_kwargs):
        super().__init__(dataloader, start, interval, by_epoch, save_best,
                         rule, test_fn, greater_keys, less_keys, out_dir,
                         file_client_args, **eval_kwargs)
        self.gts, self.pts = [], []
        self.fomo = fomo

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return

        if self.fomo:
            self.test_fn = single_gpu_test_fomo
        super()._do_evaluate(runner)

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(results,
                                                    logger=runner.logger,
                                                    fomo=self.fomo,
                                                    **self.eval_kwargs)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None