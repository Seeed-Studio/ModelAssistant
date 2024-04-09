# Copyright (c) Seeed Tech Ltd. All rights reserved.
import subprocess
from typing import Iterable, Optional, Union


def net_online() -> bool:
    """Check whether the current device is connected to the Internet."""
    import socket

    for host in '1.1.1.1', '8.8.8.8', '223.5.5.5':  # Cloudflare, Google, AliDNS:
        try:
            test_connection = socket.create_connection(address=(host, 53), timeout=2)
        except (socket.timeout, socket.gaierror, OSError):
            continue
        else:
            # If the connection was successful, close it to avoid a ResourceWarning
            test_connection.close()
            return True
    return False


def install_lib(name: Union[str, Iterable[str]], version: Optional[Union[str, Iterable[str]]] = None) -> bool:
    """Install python third-party libraries."""
    if isinstance(name, str):
        name = [name]
    if version is not None:
        if isinstance(version, str):
            name = [i + version for i in name]
        else:
            name = [i + version[idx] for idx, i in enumerate(name)]

    if version is not None:
        name = '=='.join(name)
    try:
        assert net_online(), 'Network Connection Failure!'
        print(f'Third-party library {name} being installed')
        subprocess.check_output(f'pip install --no-cache {name}', shell=True).decode()
        return True
    except Exception:
        print(f'Warning ⚠️: Installation of {name} has failed, the installation process has been skipped')
        return False


def check_lib(name: str, version: Optional[str] = None, install: bool = True) -> bool:
    """Check if the third-party libraries have been installed."""
    import pkg_resources as pkg

    flag = True
    try:
        pkg.require(name)
    except pkg.DistributionNotFound:
        try:
            import importlib

            importlib.import_module(next(pkg.parse_requirements(name)).name)
        except ImportError:
            flag = False
    except pkg.VersionConflict:
        pass

    if install and not flag:
        return install_lib(name, version=version)
    return flag
