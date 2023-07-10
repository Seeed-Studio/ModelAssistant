import json
import warnings
from typing import Optional

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms.base import BaseTransform

from edgelab.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadSensorFromFile(BaseTransform):
    """Load an Sensor sample data from file.

    Required keys:

    - "file_path": Path to the Sensor sample data file.

    Modified keys:

    - "data": Sensor sample data loaded from the file.
    - "sensor": Sensor type and unit loaded from the file.
    """

    def __init__(self, file_client_args: Optional[dict] = None, backend_args: Optional[dict] = None) -> None:
        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. ' 'Please use "backend_args" instead',
                DeprecationWarning,
            )
            if backend_args is not None:
                raise ValueError('"file_client_args" and "backend_args" cannot be set ' 'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load Axes.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['file_path']

        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(self.file_client_args, filename)
                lable_bytes = file_client.get(filename)
            else:
                lable_bytes = fileio.get(filename, backend_args=self.backend_args)
            label = json.loads(lable_bytes)
            sensors = label['payload']["sensors"]
            data = np.array([], np.float32)
            for value in label['payload']['values']:
                data = np.append(data, value)
        except Exception as e:
            raise e

        results['data'] = data
        results['sensors'] = sensors

        return results
