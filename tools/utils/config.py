import re
import os.path as osp

from mmcv.utils import Config


def load_config(filename, args=None):
    with open(filename, 'r', encoding='gb2312') as f:
        data = f.read()
    tmp_dict = {}
    exec(data, tmp_dict)
    cfg_dir = osp.dirname(filename)
    if '_base_' in tmp_dict.keys():
        base = tmp_dict['_base_']
        if isinstance(base, str):
            data = load_config(osp.join(cfg_dir, base)) + '\n' + data
        elif isinstance(base, (list, tuple)):
            for i in base:
                data = load_config(osp.join(cfg_dir, i)) + '\n' + data

    data = re.sub(f'_base_', f'old_base', data)
    if not args: return data
    for key, value in args.items():
        if isinstance(value, (int, float)):
            data = re.sub(f"^{key}\s?=\s?[^,{key}].*?[^,{key}]\n",
                          f'{key}={value}\n',
                          data,
                          flags=re.MULTILINE)
        else:
            value = value.replace('\\', '/')
            data = re.sub(f"^{key}\s?=\s?['\"]{{1}}.*?['\"]{{1}}\n",
                          f'{key}="{value}"\n',
                          data,
                          flags=re.MULTILINE)
    return data


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
                    assert not isinstance(value, (dict, list, tuple)), \
                        f'for the format of string cfg is ' \
                        f"'xxxxx${key}xxxxx' or 'xxx${key}xxx${key}xxx', " \
                        f"the type of the value of '${key}' " \
                        f'can not be dict, list, or tuple' \
                        f'but you input {type(value)} in {cfg}'
                    cfg = cfg.replace(key, str(value))
            return cfg
        else:
            return cfg

    # the pattern of string "${key}"
    pattern_key = re.compile(r'\$\{[a-zA-Z\d_.]*\}')
    # the type of ori_cfg._cfg_dict is mmcv.utils.config.ConfigDict
    updated_cfg = Config(replace_value(ori_cfg._cfg_dict),
                         cfg_text=ori_cfg._text)
    # replace the model with model_wrapper
    if updated_cfg.get('model_wrapper', None) is not None:
        updated_cfg.model = updated_cfg.model_wrapper
        updated_cfg.pop('model_wrapper')
    return updated_cfg