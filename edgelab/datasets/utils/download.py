import os
import platform
from typing import AnyStr, List


def check_compress(file):
    compress_tools = {
        'tar': "tar -xf {} -C ..",
        "gz": "gzip -d {} ..",
        "tgz": "tar -zxf {} -C ..",
        "zip": "unzip -n {} -d ..",
        "rar": "unrar e -o- -y {} ..",
    }
    if 'tar.gz' in file:
        return ["tar -zxf {} -C .."]
    fls = file.split('.')
    res = []
    for f in fls[::-1]:
        if f in compress_tools.keys():
            res.append(compress_tools[f])
    return res


def defile(files, store_dir):
    res = []
    for f in files:
        cmd = check_compress(f)
        res.append(cmd)


def download(links: List or AnyStr, store_path: AnyStr or __path__, unzip_dir=None):
    if isinstance(links, str):
        links = [links]
    os.chdir(store_path)
    if not os.path.exists('download'):
        os.mkdir('download')
    os.chdir('download')

    print(links)
    print(store_path)
    for link in links:
        file_name = link.split('/')[-1]
        unzip = check_compress(file_name)
        unzip = [de.format(file_name) for de in unzip]
        if os.path.exists(file_name):
            print(f"file {file_name} already exists in {store_path}/download")
            if not os.path.exists('.' + file_name):
                for de in unzip:
                    os.system(de)
                os.system(f'touch .{file_name}')
            continue

        # Dlownd compress dataset
        cmd = f"curl -L {link} -o {file_name} --retry 3 -C -"
        os.system(cmd)
        for de in unzip:
            os.system(de)
        os.system(f'touch .{file_name}')


def check_file(path, store_dir=None, data_name=None):
    download_dir = None
    if isinstance(path, (list, tuple)):
        if 'https://' in path[0] or 'http://' in path[0]:
            download_dir = (
                f"{os.environ['HOME']}/datasets"
                if platform.system() == 'Linux' and not store_dir
                else 'D:\datasets'
                if not store_dir
                else store_dir
            )
            download_dir = os.path.join(download_dir, data_name) if data_name else download_dir
            if not os.path.exists(download_dir):
                os.makedirs(download_dir, exist_ok=True)  # makedir the datasets

            download(path, download_dir, data_name)
    elif isinstance(path, str):
        if path.startswith('~'):
            path = path.replace('~', os.environ['HOME'])
        download_dir = path
        if 'https://' in path or 'http://' in path:
            download_dir = (
                f"{os.environ['HOME']}/datasets"
                if platform.system() == 'Linux' and not store_dir
                else 'D:\datasets'
                if not store_dir
                else store_dir
            )
            download_dir = os.path.join(download_dir, data_name) if data_name else download_dir
            if not os.path.exists(download_dir):
                os.makedirs(download_dir, exist_ok=True)  # makedir the datasets

            download(path, download_dir, data_name)

    else:
        raise TypeError(
            'The download link must be a list or a string, but got {} type'.format(
                getattr(type(path), '__name__', repr(type(path)))
            )
        )

    return download_dir
