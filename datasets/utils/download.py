import os
import platform
from typing import List, AnyStr


def check_compress(file):
    compress_tools = {
        'tar': "tar",
        "gz": "gunzip",
        "tgz": "tar",
        "zip": "unzip",
        "rar": "unrar"
    }
    if 'tar.gz' in file: return ["tar -z"]
    fls = file.split('.')
    res = []
    for f in fls[::-1]:
        if f in compress_tools.keys():
            res.append(compress_tools[f])
    return res


def defile(files, store_dir):
    res = []
    for f in file:
        cmd = check_compress(f)
        res.append(cmd)


def download(links: List or AnyStr,
             store_path: AnyStr or __path__,
             unzip_dir=None):
    if isinstance(links, list):
        pass
    elif isinstance(links, str):
        links = [links]
    else:
        raise TypeError(
            'The download link must be a list or a string, but got {} type'.
            format(getattr(type(links), '__name__', repr(type(links)))))

    os.chdir(store_path)
    print(links)
    print(store_path)
    for link in links:
        file_name = link.split('/')[-1]
        unzip = f"tar -zxvf {file_name} -C ."
        if os.path.exists(file_name):
            print(f"file {file_name} already exists in {store_path}")
            print(unzip)
            os.system(unzip)
            continue
        cmd = f"curl -L {link} -o {file_name} --retry 3 -C -"
        os.system(cmd)
        os.system(unzip)


def check_file(path, store_dir=None, data_name=None):
    if 'https://' in path or 'http://' in path:
        download_dir = f"{os.environ['HOME']}/datasets" if platform.system(
        ) == 'Linux' and not store_dir else 'D:\datasets' if not store_dir else store_dir
        download_dir = os.path.join(download_dir,
                                    data_name) if data_name else download_dir
        if not os.path.exists(download_dir):
            os.makedirs(download_dir, exist_ok=True)            # makedir the datasets

        download(path, download_dir, data_name)


    else:
        download_dir = path
    return download_dir


if __name__ == "__main__":
    # download(
    #     'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz',
    #     '/home/dq/software')
    di = check_file(
        'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz',
        '/home/dq/tmp')
    print(di)