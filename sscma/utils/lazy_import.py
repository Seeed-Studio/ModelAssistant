import subprocess
import sys
import importlib


def lazy_import(module_name: str, name_alias: str = None, name_variants: dict[str, str] = {}, index_url: str = None, install_only: bool = False):
    def validate_name(name: str):
        if not name.replace("-", "_").isidentifier():
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
    def get_arch():
        if sys.platform == "linux":
            return subprocess.check_output("uname -m", shell=True).decode().strip()
        return None
    variants_map = {
        "cpu": lambda: True,
        "aarch64": lambda: get_arch() == "aarch64",
        "armv7l": lambda: get_arch() == "armv7l",
        "x86_64": lambda: get_arch() == "x86_64",
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
                module = None
                if install_only:
                    cmd = [sys.executable, '-m', 'pip', 'show', '-q', module_name]
                    ret = subprocess.call(cmd)
                    if ret != 0:
                        raise ImportError
                else:
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
                ret = subprocess.call(cmds)
                if ret != 0:
                    print(f"Please install {package_name} manually")
                    sys.exit(ret)
                if install_only:
                    return func(*args, **kwargs)
                module = importlib.import_module(name_alias)
            globals()[name_alias] = module
            return func(*args, **kwargs)
        return wrapper
    return decorator