import subprocess
import os
import re
import sys
import time
import logging
import argparse

from subprocess import Popen, PIPE
from ubuntu_utils import cmd_result, ensure_base_env, get_job

download_links = {'miniconda': 'https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh',
                  'anaconda': 'https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh'}


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
            loger.info('retrying') if i != (retry_num-1) else None
        else:
            return True
    return False


def cmd_result(txt: str):
    cmd = os.popen(txt)
    return cmd.read().rstrip().lstrip()


def get_job(argv) -> int:
    # get nprocs, if user not specified, use max(1, nproc-2)
    job = 2
    if len(argv) <= 1:
        print('your can use `python3 {} N` to set make -j [N]'.format(argv[0]))
        nproc = cmd_result('nproc')
        if nproc is not None and len(nproc) > 0:
            job = max(int(nproc) - 2, 1)
        else:
            job = 1
    else:
        job = int(argv[1])
    return job


def test_delay(name):
    p = Popen(f'ping {name} -c 3', shell=True, encoding='utf8', stdout=PIPE)
    data = p.stdout.read()
    p.wait(5)
    if len(data):
        delay = re.findall('time=(.*?) ms', data)
        print((sum([float(i) for i in delay])/len(delay)).__round__(2))
        return sum([float(i) for i in delay])/len(delay).__round__(2)


def qure_ip():
    p = subprocess.Popen('curl http://myip.ipip.net',
                         shell=True, encoding='utf8', stdout=subprocess.PIPE)
    data = p.stdout.read()
    loger.info(data)
    if '中国' in data:
        return True
    return False


def test_network():
    delay = 1000
    domain = 'mirrors.aliyun.com'
    if qure_ip():
        mirror = {'pypi.mirrors.ustc.edu.cn': 'https://pypi.mirrors.ustc.edu.cn/simple', 'pypi.douban.com': 'http://pypi.douban.com/simple/',
                  'mirrors.aliyun.com': 'http://mirrors.aliyun.com/pypi/simple/', 'pypi.tuna.tsinghua.edu.cn': 'https://pypi.tuna.tsinghua.edu.cn/simple'}
        for i in mirror.keys():
            test = test_delay(i)
            if delay > test:
                delay = test
                domain = i
        print(mirror[domain])
        return mirror[domain]

    return False


# mirror = test_network()


def qure_gpu():
    p = Popen('nvidia-smi |grep Driver', shell=True,
              encoding='utf8', stdout=PIPE)
    data = p.stdout.read()
    print(data)
    if len(data):
        gpu = re.findall('NVIDIA-SMI (.*?) ', data)[0]
        return gpu

    print('GPU:', data)
    print(p.returncode)
    return False


def download_file(link, path):
    os.chdir(path)
    if os.path.exists(os.path.join(path, link.split('/')[-1])):
        loger.info(
            f"{link.split('/')[-1]} already exists under the {path} path")
        return
    if command(f'wget {link}'):
        loger.info(f"{link.split('/')[-1]} Downloaded")
    else:
        loger.warning(f"{link.split('/')[-1]} download failed")


def write_command(command, path):
    open(path, 'w', encoding='utf8').write(command)


def anaconda_install(now_path, conda='miniconda'):
    os.chdir(now_path)
    os.makedirs('tmp', exist_ok=True)
    if command('conda -V'):
        loger.info(
            'Your conda has been installed, skip the installation this time')
        return

    file_name = download_links[conda].split('/')[-1]
    download_file(download_links[conda], now_path)

    r = subprocess.Popen(args=f'./{file_name}', stdin=PIPE,
                         stderr=PIPE, stdout=None, shell=True, encoding='utf8')
    try:
        r.communicate('\nq\n\nyes\n\nyes\n\n')
        # r.wait(15)
    except:
        r.kill()
        loger.info('')


def cuda_install(now_path):
    os.chdir(now_path)
    if not os.path.exists('./cuda_11.2.1_460.32.03_linux.run'):
        command(
            'wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda_11.2.1_460.32.03_linux.run')

    command('wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin')
    command(
        'sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600')
    command('wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb')
    command(
        'sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb')
    command('sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub')
    command('sudo apt-get update && sudo apt-get -y install cuda')


def mmlab_install(env_name, conda_name):
    if command(f'{pip} install -r {req_path}/requirements.txt'):
        loger.info('mmlab env install succeeded!')
    if command(
            '{pip} install mmcv-full  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html'):
        loger.info('mmcv-full install succeeded!')


def torch_install(env_name, conda_name):
    if not command(f'{pip} list'):
        loger.warning(f'Environment {env_name} query failed')
        return

    p = subprocess.Popen(args=f'{pip} list | grep torch', shell=True,
                         encoding='utf8', stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    data = p.stdout.read()
    if len(data) and 'torch ' in data:
        loger.warning('Torch is has installed!')
        return

    if GPU:
        # p=subprocess.Popen(args='nvcc --version | grep release',stdout=subprocess.PIPE,encoding='utf8',shell=True)
        p = subprocess.Popen(args='nvidia-smi | grep Driver',
                             stdout=subprocess.PIPE, encoding='utf8', shell=True)
        data = p.stdout.read()
        if len(data) and 'Driver' in data:
            version = re.findall('CUDA Version: (.*?) ', data)[0]
            loger.info(f'CUDA VERSION:{version}')
        else:
            loger.info('未查询到cuda版本！')
        command(
            f'{pip} install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113')
    else:
        command(f'{pip} install torch torchvision torchaudio')


def conda_acti_env(name):
    command('conda info -e')
    time.sleep(2)
    if command(f'~/anaconda3/bin/conda activate {name}'):
        loger.info(f'The virtual environment {name} is activated')
    else:
        loger.warning(f'Virtual environment {name} activation failed')


def conda_create_env(name, version=3.8):
    p = subprocess.Popen(
        f'conda info -e', stdout=subprocess.PIPE, shell=True, encoding='utf8')
    if name in p.stdout.read():
        loger.info(f'The virtual environment {name} already exists')
        return

    p = subprocess.Popen(
        f'conda create -n {name} python=={version}', stdin=subprocess.PIPE, shell=True, encoding='utf8')
    p.communicate('y\n')

    p = subprocess.Popen(
        f'conda info -e', stdout=subprocess.PIPE, shell=True, encoding='utf8')
    if name in p.stdout.read():
        loger.info(f'The virtual environment {name} has been created')
    else:
        loger.warning(f'Failed to create virtual environment {name}')


g_jobs = 16


def install_protobuf(dep_dir) -> int:
    print('-' * 10 + 'install protobuf' + '-' * 10)

    os.chdir(dep_dir)
    if not os.path.exists('protobuf-3.20.0'):
        command(
            'wget https://github.com/protocolbuffers/protobuf/releases/download/v3.20.0/protobuf-cpp-3.20.0.tar.gz'  # noqa: E501
        )
        command('tar xvf protobuf-cpp-3.20.0.tar.gz')

    os.chdir(os.path.join(dep_dir, 'protobuf-3.20.0'))

    install_dir = os.path.join(dep_dir, 'pbinstall')
    if os.path.exists(install_dir):
        command('rm -rf {}'.format(install_dir))

    command('make clean')
    command('./configure --prefix={}'.format(install_dir))
    command('make -j {} && make install'.format(g_jobs))
    protoc = os.path.join(install_dir, 'bin', 'protoc')

    print('protoc \t:{}'.format(cmd_result('{} --version'.format(protoc))))

    command(""" echo 'export PATH={}:$PATH' >> ~/mmdeploy.env """.format(
        os.path.join(install_dir, 'bin')))
    command(
        """ echo 'export LD_LIBRARY_PATH={}:$LD_LIBRARY_PATH' >> ~/mmdeploy.env """  # noqa: E501
        .format(os.path.join(install_dir, 'lib')))

    return 0


def install_pyncnn(dep_dir):
    print('-' * 10 + 'build and install pyncnn' + '-' * 10)
    time.sleep(2)

    # generate unzip and build dir
    os.chdir(dep_dir)

    # git clone
    if not os.path.exists('ncnn'):
        command(
            'git clone --depth 1 --branch 20220729  https://github.com/tencent/ncnn && cd ncnn'  # noqa: E501
        )

    ncnn_dir = os.path.join(dep_dir, 'ncnn')
    os.chdir(ncnn_dir)

    # update submodule pybind11, gslang not required
    command('git submodule init && git submodule update python/pybind11')
    # build
    if not os.path.exists('build'):
        command('mkdir build')

    os.chdir(os.path.join(ncnn_dir, 'build'))
    command('rm -rf CMakeCache.txt')
    pb_install = os.path.join(dep_dir, 'pbinstall')
    pb_bin = os.path.join(pb_install, 'bin', 'protoc')
    pb_lib = os.path.join(pb_install, 'lib', 'libprotobuf.so')
    pb_include = os.path.join(pb_install, 'include')

    cmd = 'cmake .. '
    cmd += ' -DNCNN_PYTHON=ON '
    cmd += ' -DProtobuf_LIBRARIES={} '.format(pb_lib)
    cmd += ' -DProtobuf_PROTOC_EXECUTABLE={} '.format(pb_bin)
    cmd += ' -DProtobuf_INCLUDE_DIR={} '.format(pb_include)
    cmd += ' && make -j {} '.format(g_jobs)
    cmd += ' && make install '
    command(cmd)

    # install
    os.chdir(ncnn_dir)
    command(f'cd python && {pip} install -e .')
    ncnn_cmake_dir = os.path.join(ncnn_dir, 'build', 'install', 'lib', 'cmake',
                                  'ncnn')
    onnx_path = os.path.join(ncnn_dir, 'build', 'tools', 'bin', 'onnx')
    quan_path = os.path.join(ncnn_dir, 'build', 'tools', 'bin', 'quantize')
    caffe_path = os.path.join(ncnn_dir, 'build', 'tools', 'bin', 'caffe')
    command(f'echo export PATH={onnx_path}:\$PATH >> ~/.bashrc')
    command(f'echo export PATH={quan_path}:\$PATH >> ~/.bashrc')
    command(f'echo export PATH={caffe_path}:\$PATH >> ~/.bashrc')

    os.makedirs(ncnn_cmake_dir, exist_ok=True)
    assert (os.path.exists(ncnn_cmake_dir))
    loger.info('ncnn cmake dir \t:{}'.format(ncnn_cmake_dir))
    return ncnn_cmake_dir


def main():
    """Auto install mmdeploy with ncnn. To verify this script:

    1) use `sudo docker run -v /path/to/mmdeploy:/root/mmdeploy -v /path/to/Miniconda3-latest-Linux-x86_64.sh:/root/miniconda.sh -it ubuntu:18.04 /bin/bash` # noqa: E501
    2) install conda and setup python environment
    3) run `python3 tools/scripts/build_ubuntu_x64_ncnn.py`

    Returns:
        _type_: _description_
    """
    global g_jobs
    g_jobs = get_job(sys.argv)
    print('g_jobs {}'.format(g_jobs))

    work_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    dep_dir = os.path.abspath(os.path.join(work_dir, '..', 'mmdeploy-dep'))
    dep_dir = '/home/dq/software'
    if not os.path.exists(dep_dir):
        if os.path.isfile(dep_dir):
            loger.info(
                '{} already exists and it is a file, exit.'.format(work_dir))
            return -1
        os.mkdir(dep_dir)

    success = ensure_base_env(work_dir, dep_dir)
    if success != 0:
        return -1

    if install_protobuf(dep_dir) != 0:
        return -1

    ncnn_cmake_dir = install_pyncnn(dep_dir)


def pare_args():
    args = argparse.ArgumentParser()
    args.add_argument('--action', default='')
    args.add_argument('--envname', default='edgelab3',
                      help='conda vertual enverimen name')
    args.add_argument('--conda', default='miniconda',
                      help='conda vertual enverimen name')
    args.add_argument('--force', action='store_true',
                      help='wether force install')
    return args.parse_args()


if __name__ == '__main__':
    now_path = '/home/dq/software'
    args = pare_args()
    loger = log_init()
    GPU = qure_gpu()
    pip = f'~/{args.conda}3/envs/{args.envname}/bin/pip'

    anaconda_install(now_path, conda=args.conda)
    conda_create_env(args.envname)

    # # cuda_install(now_path)
    torch_install(args.envname, args.conda)
    self_dir = os.path.abspath(__file__)
    req_path = os.path.join(os.path.dirname(self_dir), '..', 'requirements')

    mmlab_install(args.envname, args.conda)

    install_protobuf(now_path)
    install_pyncnn(now_path)
