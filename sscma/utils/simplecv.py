import numpy as np
from typing import Union,Sequence,Tuple,List,Optional,Dict,Iterable

import cv2


import io
import os.path as osp
import warnings
from pathlib import Path

import mmengine.fileio as fileio
import numpy as np
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
from mmengine.utils import is_filepath, is_str

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None



cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}




def simplecv_imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = 'cv2'
    if backend != 'cv2':
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2'")


    resized_img = cv2.resize(
        img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale

def simplecv_imflip(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))
    

def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def simplecv_rescale_size(old_size: tuple,
                 scale: Union[float, int, Tuple[int, int]],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size




def simplecv_imrescale(
    img: np.ndarray,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    backend: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = simplecv_rescale_size((w, h), scale, return_scale=True)
    rescaled_img = simplecv_imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img

def simplecv_bbox_clip(bboxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """Clip bboxes to fit the image shape.

    Args:
        bboxes (ndarray): Shape (..., 4*k)
        img_shape (tuple[int]): (height, width) of the image.

    Returns:
        ndarray: Clipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
    cmin[0::2] = img_shape[1] - 1
    cmin[1::2] = img_shape[0] - 1
    clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
    return clipped_bboxes


def simplecv_bbox_scaling(bboxes: np.ndarray,
                 scale: float,
                 clip_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Scaling bboxes w.r.t the box center.

    Args:
        bboxes (ndarray): Shape(..., 4).
        scale (float): Scaling factor.
        clip_shape (tuple[int], optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).

    Returns:
        ndarray: Scaled bboxes.
    """
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return simplecv_bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes    



def simplecv_imcrop(
    img: np.ndarray,
    bboxes: np.ndarray,
    scale: float = 1.0,
    pad_fill: Union[float, list, None] = None
) -> Union[np.ndarray, List[np.ndarray]]:
    """Crop image patches.

    3 steps: scale the bboxes -> clip bboxes -> crop and pad.

    Args:
        img (ndarray): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale (float, optional): Scale ratio of bboxes, the default value
            1.0 means no scaling.
        pad_fill (Number | list[Number]): Value to be filled for padding.
            Default: None, which means no padding.

    Returns:
        list[ndarray] | ndarray: The cropped image patches.
    """
    chn = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert len(pad_fill) == chn

    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = simplecv_bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = simplecv_bbox_clip(scaled_bboxes, img.shape)

    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
        if pad_fill is None:
            patch = img[y1:y2 + 1, x1:x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            patch_h = _y2 - _y1 + 1
            patch_w = _x2 - _x1 + 1
            if chn == 1:
                patch_shape = (patch_h, patch_w)
            else:
                patch_shape = (patch_h, patch_w, chn)  # type: ignore
            patch = np.array(
                pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start:y_start + h, x_start:x_start + w,
                  ...] = img[y1:y1 + h, x1:x1 + w, ...]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches


def simplecv_imread(img_or_path: Union[np.ndarray, str, Path],
           flag: str = 'color',
           channel_order: str = 'bgr',
           backend: Optional[str] = None,
           file_client_args: Optional[dict] = None,
           *,
           backend_args: Optional[dict] = None) -> np.ndarray:
    """Read an image.

    Args:
        img_or_path (ndarray or str or Path): Either a numpy array or str or
            pathlib.Path. If it is a numpy array (loaded image), then
            it will be returned as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale`, `unchanged`,
            `color_ignore_orientation` and `grayscale_ignore_orientation`.
            By default, `cv2` and `pillow` backend would rotate the image
            according to its EXIF info unless called with `unchanged` or
            `*_ignore_orientation` flags. `turbojpeg` and `tifffile` backend
            always ignore image's EXIF info regardless of the flag.
            The `turbojpeg` backend only supports `color` and `grayscale`.
        channel_order (str): Order of channel, candidates are `bgr` and `rgb`.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`.
            If backend is None, the global imread_backend specified by
            ``mmcv.use_backend()`` will be used. Default: None.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Default: None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> import mmcv
        >>> img_path = '/path/to/img.jpg'
        >>> img = mmcv.imread(img_path)
        >>> img = mmcv.imread(img_path, flag='color', channel_order='rgb',
        ...     backend='cv2')
        >>> img = mmcv.imread(img_path, flag='color', channel_order='bgr',
        ...     backend='pillow')
        >>> s3_img_path = 's3://bucket/img.jpg'
        >>> # infer the file backend by the prefix s3
        >>> img = mmcv.imread(s3_img_path)
        >>> # manually set the file backend petrel
        >>> img = mmcv.imread(s3_img_path, backend_args={
        ...     'backend': 'petrel'})
        >>> http_img_path = 'http://path/to/img.jpg'
        >>> img = mmcv.imread(http_img_path)
        >>> img = mmcv.imread(http_img_path, backend_args={
        ...     'backend': 'http'})
    """
    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead', DeprecationWarning)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                'same time.')

    if isinstance(img_or_path, Path):
        img_or_path = str(img_or_path)

    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif is_str(img_or_path):
        if file_client_args is not None:
            file_client = fileio.FileClient.infer_client(
                file_client_args, img_or_path)
            img_bytes = file_client.get(img_or_path)
        else:
            img_bytes = fileio.get(img_or_path, backend_args=backend_args)
        return simplecv_imfrombytes(img_bytes, flag, channel_order, backend)
    else:
        raise TypeError('"img" must be a numpy array or a str or '
                        'a pathlib.Path object')


def simplecv_imfrombytes(content: bytes,
                flag: str = 'color',
                channel_order: str = 'bgr',
                backend: Optional[str] = None) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`. If backend is
            None, the global imread_backend specified by ``mmcv.use_backend()``
            will be used. Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> img_path = '/path/to/img.jpg'
        >>> with open(img_path, 'rb') as f:
        >>>     img_buff = f.read()
        >>> img = mmcv.imfrombytes(img_buff)
        >>> img = mmcv.imfrombytes(img_buff, flag='color', channel_order='rgb')
        >>> img = mmcv.imfrombytes(img_buff, backend='pillow')
        >>> img = mmcv.imfrombytes(img_buff, backend='cv2')
    """

    if backend is None:
        backend = 'cv2'
    if backend != 'cv2' :
        raise ValueError(
            f'backend: {backend} is not supported. Supported '
            "backends are 'cv2''")

    img_np = np.frombuffer(content, np.uint8)
    flag = imread_flags[flag] if is_str(flag) else flag
    img = cv2.imdecode(img_np, flag)
    if flag == IMREAD_COLOR and channel_order == 'rgb':
         cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    return img


def simplecv_imwrite(img: np.ndarray,
            file_path: str,
            params: Optional[list] = None,
            auto_mkdir: Optional[bool] = None,
            file_client_args: Optional[dict] = None,
            *,
            backend_args: Optional[dict] = None) -> bool:
    """Write image to file.

    Warning:
        The parameter `auto_mkdir` will be deprecated in the future and every
        file clients will make directory automatically.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically. It will be deprecated.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Default: None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.

    Returns:
        bool: Successful or not.

    Examples:
        >>> # write to hard disk client
        >>> ret = mmcv.imwrite(img, '/path/to/img.jpg')
        >>> # infer the file backend by the prefix s3
        >>> ret = mmcv.imwrite(img, 's3://bucket/img.jpg')
        >>> # manually set the file backend petrel
        >>> ret = mmcv.imwrite(img, 's3://bucket/img.jpg', backend_args={
        ...     'backend': 'petrel'})
    """
    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead', DeprecationWarning)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                'same time.')

    assert is_filepath(file_path)
    file_path = str(file_path)
    if auto_mkdir is not None:
        warnings.warn(
            'The parameter `auto_mkdir` will be deprecated in the future and '
            'every file clients will make directory automatically.')

    img_ext = osp.splitext(file_path)[-1]
    # Encode image according to image suffix.
    # For example, if image path is '/path/your/img.jpg', the encode
    # format is '.jpg'.
    flag, img_buff = cv2.imencode(img_ext, img, params)

    if file_client_args is not None:
        file_client = fileio.FileClient.infer_client(file_client_args,
                                                     file_path)
        file_client.put(img_buff.tobytes(), file_path)
    else:
        fileio.put(img_buff.tobytes(), file_path, backend_args=backend_args)

    return flag

