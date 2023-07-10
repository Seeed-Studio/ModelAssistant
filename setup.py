import os
import platform
import shutil
import subprocess
import sys
import warnings
from distutils import spawn

from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'edgelab/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_line(line: str):
    if '#' in line:
        return line.split('#')[0]
    else:
        return line


def get_cuda_availability():
    nvidia_smi = 'nvidia-smi'
    if platform.system() == 'Windows':
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = f"{os.environ['systemdrive']}\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe"
    try:
        subprocess.check_output(nvidia_smi)
    except Exception:
        return False
    return True


def parse_requirements(fpath: str = ''):
    reqs = []
    index = []
    if fpath.strip() == '':
        if 'install' not in sys.argv:
            return reqs, index
        if get_cuda_availability():
            fpath = 'requirements_cuda.txt'
        else:
            fpath = 'requirements.txt'
    with open(fpath, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('-r'):
                    res = parse_requirements(line.split(' ')[-1])
                    reqs += res[0]
                    index += res[1]
                elif line.startswith('-i'):
                    index.append(parse_line(line))
                else:
                    reqs.append(parse_line(line))
    return reqs, index


def add_mim_extension():
    if platform.system() == 'Windows':
        mode = 'copy'
    else:
        mode = 'symlink'
    if 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        mode = 'copy'

    filenames = ['tools', 'configs']
    repo_path = os.path.dirname(__file__)
    tools_path = os.path.join(repo_path, 'edgelab')
    os.makedirs(tools_path, exist_ok=True)

    for filename in filenames:
        if os.path.exists(filename):
            src_path = os.path.join(repo_path, filename)
            tar_path = os.path.join(tools_path, filename)

            if os.path.isfile(tar_path) or os.path.islink(tar_path):
                os.remove(tar_path)
            elif os.path.isdir(tar_path):
                shutil.rmtree(tar_path)

            if mode == 'symlink':
                src_relpath = os.path.relpath(src_path, os.path.dirname(tar_path))
                os.symlink(src_relpath, tar_path)
            elif mode == 'copy':
                if os.path.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif os.path.isdir(src_path):
                    shutil.copytree(src_path, tar_path, dirs_exist_ok=True)
                else:
                    warnings.warn(f'Cannot copy file {src_path}.')
            else:
                raise ValueError(f'Invalid mode {mode}')


if __name__ == '__main__':
    requirements, index_urls = parse_requirements()
    add_mim_extension()
    setup(
        name='edgelab',
        version=get_version(),
        description='Seeed Studio EdgeLab is an open-source project focused on embedded AI. \
            We optimize the excellent algorithms from OpenMMLab for real-world scenarios and make implementation more user-friendly,\
             achieving faster and more accurate inference on embedded devices.',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='EdgeLab Contributors',
        author_email='',
        maintainer='EdgeLab Contributors',
        keywords='embedded AI',
        url='https://www.seeedstudio.com/',
        download_url='https://github.com/Seeed-Studio/EdgeLab',
        packages=find_packages(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: Unix',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
        ],
        install_requires=requirements,
        dependency_links=index_urls,
        include_package_data=True,
        python_requires='>=3.8',
        license='MIT',
        entry_points={
            'console_scripts': [
                'edgtrain=edgelab.tools.train:main',
                'edginfer=edgelab.tools.test:main',
                'edgenv=edgelab.tools.env_config:main',
                'edgexport=edgelab.tools.export:main',
                'edg2onnx=edgelab.tools.torch2onnx:main',
            ]
        },
    )
