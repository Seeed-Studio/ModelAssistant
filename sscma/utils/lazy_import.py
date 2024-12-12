import subprocess
import sys
import importlib


def lazy_import(module_name: str, name_alias: str = None, name_variants: dict[str, str] = {}, index_url: str = None):
    def validate_name(name: str):
        if not name.isidentifier():
            raise ValueError(f"Invalid module name: {name}")
    if name_alias is None:
        validate_name(module_name)
    else:
        validate_name(name_alias)
    def has_cuda():
        try:
            from torch.cuda import is_available as _is_torch_cuda_available
            return _is_torch_cuda_available()
        except ImportError:
            if sys.platform == "linux":
                return subprocess.check_call(
                    "nvidia-smi", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ) == 0
    variants_map = {
        "cpu": lambda: True,
        "cuda": has_cuda,
        "rocm": lambda: False,
    }
    name_alias = name_alias or module_name
    def decorator(func):
        def wrapper(*args, **kwargs):
            if globals().get(name_alias) is not None:
                print(f"Warning: {name_alias} already exists in the global scope, skipped")
                return func(*args, **kwargs)
            try:
                module = importlib.import_module(name_alias)
            except ImportError:
                package_name = module_name
                for n, k in name_variants.items():
                    c = variants_map.get(n)
                    if c is not None and c():
                        package_name = k
                        break
                cmds = [sys.executable, '-m', 'pip', 'install']
                cmds.append(package_name)
                if index_url is not None:
                    cmds.extend(['-i', index_url])
                ret = subprocess.check_call(cmds)
                if ret != 0:
                    print(f"Please install {package_name} manually")
                    sys.exit(ret)
                module = importlib.import_module(name_alias)
            globals()[name_alias] = module
            return func(*args, **kwargs)
        return wrapper
    return decorator