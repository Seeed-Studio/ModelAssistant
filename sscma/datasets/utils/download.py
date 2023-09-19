import re
import os
from tqdm.std import tqdm
import platform
from typing import AnyStr, List, Optional


def download_file(url, path) -> bool:
    try:
        import requests
    except ImportError:
        ImportError(
            'An error occurred importing the "requests" module, please execute "pip install requests" and try again.'
        )
    try:
        response = requests.get(url, stream=True)
        if os.path.exists(path) and int(response.headers.get("content-length")) == os.path.getsize(path):
            print("File A already exists, skip download!!!")
            return
        # write the zip file to the desired path
        with open(path, "wb") as f:
            total_length = int(response.headers.get("content-length"))
            for chunk in tqdm(
                response.iter_content(chunk_size=1024),
                desc=f"Downloading file to {path}",
                total=int(total_length / 1024) + 1,
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
    except Exception as e:
        print(f'An error occurred downloading the file, please check if the link "{url}" is correct.')
        raise e


def _download(url: str, api_key: Optional[str] = None) -> str:
    """
    roboflow dataset downloader.The api_key can be set through the environment variable API_KEY.
    For how to get the value of api_key you can see the following link:
        https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key

    Param:
        url(str): Links to the corresponding roboflow datasets (including
            web links and packaged data links)
        api_key(str): API key for roboflow, TODO

    Return:
        extrapath(str): The path of the unzipped file after downloading the
            dataset.

    """
    import zipfile

    if not is_link(url):
        raise ValueError(f'Checked "{url}" is not a link, please check the download link and try again.')

    if api_key is None:
        api_key = os.getenv("API_KEY", None)

    # check_roboflow_link() TODO

    if api_key is None:
        # download link
        pwd = os.getcwd()
        extrapath = os.path.join(pwd, url.split('=')[-1]) + os.path.sep
        zippath = os.path.join(pwd, url.split('=')[-1]) + '.zip'
        download_file(url, zippath)
        if os.path.exists(extrapath):
            print("The compressed file has been extracted, skip the decompression!!!")
            return extrapath
        os.makedirs(extrapath, exist_ok=True)
        with zipfile.ZipFile(zippath, "r") as zip_ref:
            for member in tqdm(
                zip_ref.infolist(),
                desc="Extracting Dataset",
            ):
                try:
                    zip_ref.extract(member, extrapath)
                except zipfile.error:
                    raise RuntimeError("Error unzipping download")
        return extrapath
    else:
        raise RuntimeError("An error occurred while executing the file download, please check if the link is correct.")


def is_link(url: str) -> bool:
    pattern = r'^(http|https|ftp)://[^\s/$.?#].[^\s]*$'
    match = re.match(pattern, url)
    return bool(match)


def check_compress(file):
    compress_tools = {
        'tar': 'tar -xf {} -C ..',
        'gz': 'gzip -d {} ..',
        'tgz': 'tar -zxf {} -C ..',
        'zip': 'unzip -n {} -d ..',
        'rar': 'unrar e -o- -y {} ..',
    }
    if 'tar.gz' in file:
        return ['tar -zxf {} -C ..']
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


def download(links: List or AnyStr, store_path: AnyStr, unzip_dir=None):
    if isinstance(links, str):
        links = [links]
    os.chdir(store_path)
    if not os.path.exists('download'):
        os.mkdir('download')
    os.chdir('download')

    for link in links:
        file_name = link.split('/')[-1]
        unzip = check_compress(file_name)
        unzip = [de.format(file_name) for de in unzip]
        if os.path.exists(file_name):
            print(f'file {file_name} already exists in {store_path}/download')
            if not os.path.exists('.' + file_name):
                for de in unzip:
                    os.system(de)
                os.system(f'touch .{file_name}')
            continue

        # Dlownd compress dataset
        cmd = f'curl -L {link} -o {file_name} --retry 3 -C -'
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
