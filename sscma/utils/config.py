import os.path as osp
import re
from typing import Optional, Sequence, Union

from mmengine.config import Config
from sscma.datasets.utils.download import is_link, _download


def dump_config_to_log_dir(self) -> None:
    """Dump config to `work_dir`."""
    if self.cfg.filename is not None:
        filename = osp.basename(self.cfg.filename)
    else:
        filename = f'{self.timestamp}.py'
    self.cfg.dump(osp.join(self.log_dir, filename))


def replace(data: str, args: Optional[dict] = None) -> str:
    """Replace the basic configuration items in the configuration file.

    Args:
        data(str): the string to be replaced
        args(dict): the replaced value

    Returns:
        data(str): the replaced string
    """
    if not args:
        return data
    for key, value in args.items():
        if isinstance(value, (int, float)):
            data = re.sub(f'^{key}\s?=\s?[^,{key}].*?[^,{key}].*?$\n', f'{key}={value}\n', data, flags=re.MULTILINE)
        elif isinstance(value, (list, tuple)):
            data = re.sub(f"^{key}\s?=\s?['\"]{{1}}.*?['\"]{{1}}.*?$\n", f'{key}="{value}"\n', data, flags=re.MULTILINE)
        elif isinstance(value, str):
            value = value.replace('\\', '/')
            data = re.sub(f"^{key}\s?=\s?['\"]{{1}}.*?['\"]{{1}}.*?$\n", f'{key}="{value}"\n', data, flags=re.MULTILINE)
    return data


def replace_base_(data: str, base: Union[str, Sequence[str]]) -> str:
    """Replace the _base_ configuration item in the configuration file.

    Args:
        data(str): the string to be replaced
        base(str|[str]): the replaced value

    Returns:
        data(str): the replaced string
    """
    if isinstance(base, str):
        pattern = "_base_\s?=\s?[',\"].+?[',\"]"
        data = re.sub(pattern, f"_base_ = '{base}'", data, flags=re.MULTILINE)
    elif isinstance(base, (list, tuple)):
        pattern = '_base_\s?=\s?[\[].+?[\]]'
        data = re.sub(pattern, f'_base_ = {str(base)}', data, flags=re.S)

    return data


def load_config(filename: str, folder: str, cfg_options: Optional[dict] = None) -> str:
    """Load the configuration file and modify the value in cfg-options at the
    same time, write the modified file to the temporary file, and finally store
    the modified file in the temporary path and return the corresponding path.

    Args:
        filename: configuration file path
        cfg_options: Parameters passed on the command line to modify the
            configuration file
        folder: The path to the temporary folder, all temporary files in
            the function will be stored in this folder

    Returns:
        cfg_path: The path of the replaced temporary file is equal to the
            path of the corresponding file after filename is modified
    """
    with open(filename, 'r', encoding='gb2312') as f:
        data = f.read()
        tmp_dict = {}
        exec(data, tmp_dict)
        if cfg_options is None:
            cfg_options = {}

        if cfg_options is not None and cfg_options.get("data_root", ''):
            if is_link(cfg_options.get("data_root")):
                data_root = _download(cfg_options.get("data_root"))
                cfg_options['data_root'] = data_root
        elif 'data_root' in tmp_dict and is_link(tmp_dict.get("data_root")):
            data_root = _download(tmp_dict.get("data_root"))
            cfg_options['data_root'] = data_root

        data = replace(data, cfg_options)

    tmp_dict = {}
    exec(data, tmp_dict)
    cfg_dir = osp.dirname(filename)
    cfg_file = osp.basename(filename)

    if '_base_' in tmp_dict.keys():
        base = tmp_dict['_base_']
        if isinstance(base, str):
            _base_path = load_config(
                osp.join(cfg_dir, base),
                folder=folder,
                cfg_options=cfg_options,
            )
            data = replace_base_(data, _base_path)

        elif isinstance(base, (tuple, list)):
            _tmp_base = []
            for _base in base:
                _base_path = load_config(
                    osp.join(cfg_dir, _base),
                    folder=folder,
                    cfg_options=cfg_options,
                )
                _tmp_base.append(_base_path)

            data = replace_base_(data, _tmp_base)
    p = osp.join(folder, cfg_file)
    with open(p, 'w', encoding='gb2312') as f:
        f.write(data)

    return p


def replace_cfg_vals(ori_cfg):
    """Replace the string "${key}" with the corresponding value.

    Replace the "${key}" with the value of ori_cfg.key in the config. And
    support replacing the chained ${key}. Such as, replace "${key0.key1}"
    with the value of cfg.key0.key1. Code is modified from `vars.py
    < https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/vars.py>`_  # noqa: E501

    Args:
        ori_cfg (mmcv.utils.config.Config):
            The origin config with "${key}" generated from a file.

    Returns:
        updated_cfg [mmcv.utils.config.Config]:
            The config with "${key}" replaced by the corresponding value.
    """

    def get_value(cfg, key):
        for k in key.split('.'):
            cfg = cfg[k]
        return cfg

    def replace_value(cfg):
        if isinstance(cfg, dict):
            return {key: replace_value(value) for key, value in cfg.items()}
        elif isinstance(cfg, list):
            return [replace_value(item) for item in cfg]
        elif isinstance(cfg, tuple):
            return tuple([replace_value(item) for item in cfg])
        elif isinstance(cfg, str):
            # the format of string cfg may be:
            # 1) "${key}", which will be replaced with cfg.key directly
            # 2) "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx",
            # which will be replaced with the string of the cfg.key
            keys = pattern_key.findall(cfg)
            values = [get_value(ori_cfg, key[2:-1]) for key in keys]
            if len(keys) == 1 and keys[0] == cfg:
                # the format of string cfg is "${key}"
                cfg = values[0]
            else:
                for key, value in zip(keys, values):
                    # the format of string cfg is
                    # "xxx${key}xxx" or "xxx${key1}xxx${key2}xxx"
                    assert not isinstance(value, (dict, list, tuple)), (
                        f'for the format of string cfg is '
                        f"'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', "
                        f"the type of the value of '${key}' "
                        f'can not be dict, list, or tuple'
                        f'but you input {type(value)} in {cfg}'
                    )
                    cfg = cfg.replace(key, str(value))
            return cfg
        else:
            return cfg

    # the pattern of string "${key}"
    pattern_key = re.compile(r'\$\{[a-zA-Z\d_.]*\}')
    # the type of ori_cfg._cfg_dict is mmcv.utils.config.ConfigDict
    updated_cfg = Config(replace_value(ori_cfg._cfg_dict), cfg_text=ori_cfg._text)
    # replace the model with model_wrapper
    if updated_cfg.get('model_wrapper', None) is not None:
        updated_cfg.model = updated_cfg.model_wrapper
        updated_cfg.pop('model_wrapper')
    return updated_cfg
