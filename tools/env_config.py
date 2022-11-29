import os
import re
import time
import logging
import argparse
import subprocess
import os.path as osp

from subprocess import Popen, PIPE
from ubuntu_utils import cmd_result, ensure_base_env

download_links = {
    'miniconda':
    'https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh',
    'anaconda':
    'https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh'
}


def log_init():
    loger = logging.getLogger('ENV_CONFIG')
    loger.setLevel(logging.INFO)

    hd = logging.StreamHandler()
    hd.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    hd.setFormatter(formatter)
    loger.addHandler(hd)
    return loger


def command(cmd, retry_num=3):
    for i in range(retry_num):
        if os.system(cmd) != 0:
            time.sleep(1)
            loger.warning(f'COMMAND:"{cmd}"execution failed')
            loger.info('retrying') if i != (retry_num - 1) else None
        else:
            return True
    return False


def test_delay(name):
    p = Popen(f'ping {name} -c 3', shell=True, encoding='utf8', stdout=PIPE)
    loger.info(f'testing to "{name}" delay')
    data = p.stdout.read()
    p.wait(5)
    if len(data):
        delay = re.findall('time=(.*?) ms', data)
        return sum([float(i) for i in delay]) / len(delay).__round__(2) if len(
            delay) else 1000


def qure_ip():
    p = subprocess.Popen('curl http://myip.ipip.net',
                         shell=True,
                         encoding='utf8',
                         stdout=subprocess.PIPE)
    data = p.stdout.read()
    loger.info(data)
    if '中国' in data:
        return True
    return False


def test_network():
    delay = 1000
    domain = 'pypi.tuna.tsinghua.edu.cn'
    if qure_ip():
        mirror = {
            'pypi.mirrors.ustc.edu.cn':
            'https://pypi.mirrors.ustc.edu.cn/simple',
            'pypi.douban.com': 'http://pypi.douban.com/simple/',
            'mirrors.aliyun.com': 'https://mirrors.aliyun.com/pypi/simple/',
            'pypi.tuna.tsinghua.edu.cn':
            'https://pypi.tuna.tsinghua.edu.cn/simple'
        }
        for i in mirror.keys():
            test = test_delay(i)
            if delay > test:
                delay = test
                domain = i
        loger.info(mirror[domain])
        return mirror[domain], domain

    return False


def qure_gpu():
    p = subprocess.Popen(args='lspci |grep -i nvidia',
                         stdout=subprocess.PIPE,
                         encoding='utf8',
                         shell=True)
    data = p.stdout.read()
    if 'NVIDIA' in data.upper():
        return True
    return False


def download_file(link, path):
    os.chdir(path)
    if osp.exists(osp.join(path, link.split('/')[-1])):
        loger.info(
            f"{link.split('/')[-1]} already exists under the {path} path")
        return
    if command(f'wget {link}'):
        loger.info(f"{link.split('/')[-1]} Downloaded")
    else:
        loger.warning(f"{link.split('/')[-1]} download failed")


def anaconda_install(conda='miniconda'):
    os.chdir(proce_path)
    if command(f'{conda_bin} -V', 1):
        loger.info(
            'Your conda has been installed, skip the installation this time')
        return

    file_name = download_links[conda].split('/')[-1]
    if not osp.exists(file_name):
        download_file(download_links[conda], proce_path)

    time.sleep(1)
    command(f'chmod +x {file_name}')
    command(f'./{file_name} -b')

    # r = subprocess.Popen(args=f'./{file_name}',
    #                      stdin=PIPE,
    #                      stderr=PIPE,
    #                      stdout=None,
    #                      shell=True,
    #                      encoding='utf8')
    # os.popen('ls').writelines(['\n','q','yes\n'])
    # try:
    #     r.communicate('\nq\n\nyes\n\nyes\n\n')
    #     # r.wait(15)
    # except:
    #     r.kill()
    #     loger.info('anaconda installation failed!')


def cuda_install():
    os.chdir(proce_path)
    if not osp.exists('./cuda_11.2.1_460.32.03_linux.run'):
        command(
            'wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_460.32.03_linux.run'
        )

    command(
        'wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin'
    )
    command(
        'sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600'
    )
    command(
        'wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb'
    )
    command(
        'sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb'
    )
    command(
        'sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub')
    command('sudo apt-get update && sudo apt-get -y install cuda')


def conda_create_env(name, version=3.8):
    command(f"{conda_bin} init")
    if command(f"{pip} -V", 1):
        loger.info(f'The virtual environment {name} already exists')
        return

    p = subprocess.Popen(f'{conda_bin} create -n {name} python=={version}',
                         stdin=subprocess.PIPE,
                         shell=True,
                         encoding='utf8')
    p.communicate('y\n')

    p = subprocess.Popen(f'{conda_bin} info -e',
                         stdout=subprocess.PIPE,
                         shell=True,
                         encoding='utf8')
    if name in p.stdout.read():
        loger.info(f'The virtual environment {name} has been created')
    else:
        loger.warning(f'Failed to create virtual environment {name}')


def conda_acti_env(name):
    command(f'{conda_bin} info -e')
    time.sleep(2)
    if command(f'{conda_bin} activate {name}'):
        loger.info(f'The virtual environment {name} is activated')
    else:
        loger.warning(f'Virtual environment {name} activation failed')


def torch_install():
    p = subprocess.Popen(args=f'{pip} list | grep torch',
                         shell=True,
                         encoding='utf8',
                         stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE)
    data = p.stdout.read()
    if len(data) and 'torch ' in data:
        loger.warning('Torch is has installed!')
        return

    if GPU:
        p = subprocess.Popen(args='nvidia-smi | grep Driver',
                             stdout=subprocess.PIPE,
                             encoding='utf8',
                             shell=True)
        data = p.stdout.read()
        if len(data) and 'Driver' in data:
            version = re.findall('CUDA Version: (.*?) ', data)[0]
            loger.info(f'CUDA VERSION:{version}')
        else:
            loger.warning('cuda version not found！')
        command(
            f'{pip} install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113'
            + pip_mirror)
    else:
        command(
            f'{pip} install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0 '
            + pip_mirror)


def mmlab_install():
    req_path = osp.join(project_path, 'requirements')
    if command(f'{pip} install -r {req_path}/requirements.txt' + pip_mirror):
        loger.info('mmlab env install succeeded!')
    if GPU:
        if command(
                f'{pip} install mmcv-full  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html'
        ):
            loger.info('mmcv-full install succeeded!')
    else:
        command(f'{pip} install mmcv-full' + pip_mirror)
        loger.info('mmcv-full install succeeded!')


def install_protobuf() -> int:
    print('-' * 10 + 'install protobuf' + '-' * 10)

    os.chdir(proce_path)
    if not osp.exists('protobuf-cpp-3.20.0.tar.gz'):
        command(
            'wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.0/protobuf-cpp-3.20.0.tar.gz'  # noqa: E501
        )
    command('tar xvf protobuf-cpp-3.20.0.tar.gz')

    os.chdir(osp.join(proce_path, 'protobuf-3.20.0'))

    install_dir = osp.join(proce_path, 'pbinstall')
    if osp.exists(install_dir):
        command('rm -rf {}'.format(install_dir))
    else:
        os.makedirs(install_dir, exist_ok=True)

    command('make clean')
    command('./configure --prefix={}'.format(install_dir))
    command('make -j {} && make install'.format(g_jobs))
    protoc = osp.join(install_dir, 'bin', 'protoc')

    print('protoc \t:{}'.format(cmd_result('{} --version'.format(protoc))))

    command(""" echo 'export PATH={}:$PATH' >> ~/mmdeploy.env """.format(
        osp.join(install_dir, 'bin')))
    command(
        """ echo 'export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH' >> ~/mmdeploy.env """  # noqa: E501
        .format(osp.join(install_dir, 'lib')))

    return 0


def install_pyncnn():
    print('-' * 10 + 'build and install pyncnn' + '-' * 10)
    time.sleep(2)

    # generate unzip and build dir
    os.chdir(proce_path)

    # git clone
    if not osp.exists('ncnn'):
        command(
            'git clone --depth 1 --branch 20220729  https://github.com/tencent/ncnn && cd ncnn'  # noqa: E501
        )

    ncnn_dir = osp.join(proce_path, 'ncnn')
    os.chdir(ncnn_dir)

    # update submodule pybind11, gslang not required
    command('git submodule init && git submodule update python/pybind11')
    # build
    if not osp.exists('build'):
        command('mkdir build')

    os.chdir(osp.join(ncnn_dir, 'build'))
    command('rm -rf CMakeCache.txt')
    pb_install = osp.join(proce_path, 'pbinstall')
    pb_bin = osp.join(pb_install, 'bin', 'protoc')
    pb_lib = osp.join(pb_install, 'lib', 'libprotobuf.so')
    pb_include = osp.join(pb_install, 'include')

    cmd = 'cmake .. '
    cmd += ' -DNCNN_PYTHON=ON '
    cmd += ' -DProtobuf_LIBRARIES={} '.format(pb_lib)
    cmd += ' -DProtobuf_PROTOC_EXECUTABLE={} '.format(pb_bin)
    cmd += ' -DProtobuf_INCLUDE_DIR={} '.format(pb_include)
    cmd += ' && make -j {} '.format(g_jobs)
    cmd += ' && make install '
    command(cmd)

    # install
    os.chdir(project_path)
    command(f'{pip} install ncnn' + pip_mirror)

    path_ls = []
    path_ls.append(osp.join(ncnn_dir, 'build', 'tools', 'onnx'))
    path_ls.append(osp.join(ncnn_dir, 'build', 'tools', 'quantize'))
    path_ls.append(osp.join(ncnn_dir, 'build', 'tools', 'caffe'))
    path_ls.append(osp.join(ncnn_dir, 'build', 'install', 'bin'))
    path_ls.append(osp.join(ncnn_dir, 'build', 'install', 'include'))
    path_ls.append(osp.join(ncnn_dir, 'build', 'install', 'lib'))
    PATH = os.environ['PATH']
    for p in path_ls:
        if p in PATH: continue
        else:
            command(f'echo export PATH={p}:\$PATH >> ~/.bashrc') if osp.exists(
                p) else None


def proto_ncnn_install():
    """refer:https://github.com/open-mmlab/mmdeploy/blob/master/tools/scripts/build_ubuntu_x64_ncnn.py
    Returns:
        _type_: _description_
    """
    if success != 0:
        return -1

    if install_protobuf() != 0:
        return -1

    install_pyncnn()


def check_env():
    check_list = {}
    ncnn_dir = osp.join(proce_path, 'ncnn')
    ncnn = osp.join(ncnn_dir, 'build', 'install', 'bin', 'onnx2ncnn')
    check_list['anaconda'] = 'OK' if command(f"{conda_bin} -V", 1) else 'faile'
    check_list["virtual env"] = 'OK' if command(f"{pip} -V", 1) else 'faile'
    check_list['python ncnn'] = 'OK' if command(
        f"{python_bin} -c 'import ncnn'", 1) else 'faile'

    check_list['ncnn'] = 'OK' if osp.exists(ncnn) else 'faile'

    check_list['torch'] = 'OK' if command(f"{python_bin} -c 'import torch'",
                                          1) else 'faile'
    check_list['torchvision'] = 'OK' if command(
        f"{python_bin} -c 'import torchvision'", 1) else 'faile'
    check_list['torchaudio'] = 'OK' if command(
        f"{python_bin} -c 'import torchaudio'", 1) else 'faile'

    check_list['mmcv'] = 'OK' if command(f"{python_bin} -c 'import mmcv'",
                                         1) else 'faile'
    check_list['mmdet'] = 'OK' if command(f"{python_bin} -c 'import mmdet'",
                                          1) else 'faile'
    check_list['mmcls'] = 'OK' if command(f"{python_bin} -c 'import mmcls'",
                                          1) else 'faile'
    check_list['mmpose'] = 'OK' if command(f"{python_bin} -c 'import mmpose'",
                                           1) else 'faile'

    w, h = os.get_terminal_size()
    w0 = w - 10
    for key, value in check_list.items():
        print(f"{key:30s}:{value:10s}")


def pare_args():
    args = argparse.ArgumentParser()
    args.add_argument('--envname',
                      default='edgelab',
                      help='conda vertual enverimen name')
    args.add_argument('--conda',
                      default='anaconda',
                      help='conda vertual enverimen name')
    return args.parse_args()


def prepare():
    global args, project_path, pip, python_bin, conda_bin, home, proce_path, loger, pip_mirror, GPU, g_jobs, success
    args = pare_args()
    g_jobs = os.cpu_count() if os.cpu_count else 8
    project_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
    pip = f'~/{args.conda}3/envs/{args.envname}/bin/pip'
    home = os.environ['HOME']
    python_bin = f'~/{args.conda}3/envs/{args.envname}/bin/python'

    conda_bin = f'{home}/{args.conda}3/bin/conda'
    proce_path = f'{home}/software'
    try:
        PATH = os.environ['PYTHONPATH']
    except:
        PATH = ''
    finally:
        if proce_path not in PATH:
            command(f'echo export PYTHONPATH={home}:\$PYTHONPATH >> ~/.bashrc')

    success = ensure_base_env(project_path, proce_path)
    os.makedirs(proce_path, exist_ok=True)
    loger = log_init()
    mirror = test_network()
    pip_mirror = f' -i {mirror[0]} --trusted-host {mirror[1]}' if mirror else ''
    GPU = qure_gpu()


def main():
    prepare()

    anaconda_install(conda=args.conda)
    conda_create_env(args.envname)

    # cuda_install()
    torch_install()
    # mmlab
    mmlab_install()
    # export
    proto_ncnn_install()

    check_env()
    # command('source ~/.bashrc')


if __name__ == '__main__':
    main()