import os
import sys
import shutil
import warnings
import platform
import os.path as osp

from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'edgelab/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_line(line: str):
    if '#' in line:
        return line.split('#')[0]
    else:
        return line


def get_require(fpath='./requirements.txt'):
    res = []
    with open(fpath, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('-r'):
                    res += get_require(line.split(' ')[-1])
                else:
                    res.append(parse_line(line))

    return res


def add_mim_extension():
    """
    These files will be added by creating a symlink to the originals if the
    package is installed in `editable` mode (e.g. pip install -e .), or by
    copying from the originals otherwise.
    """

    # parse installment mode
    if 'develop' in sys.argv:
        # installed by `pip install -e .`
        if platform.system() == 'Windows':
            # set `copy` mode here since symlink fails on Windows.
            mode = 'copy'
        else:
            mode = 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        # installed by `pip install .`
        # or create source distribution by `python setup.py sdist`
        mode = 'copy'
    else:
        return

    filenames = ['tools', 'configs']
    repo_path = osp.dirname(__file__)
    tools_path = osp.join(repo_path, 'edgelab')
    os.makedirs(tools_path, exist_ok=True)

    for filename in filenames:
        if osp.exists(filename):
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(tools_path, filename)

            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)

            if mode == 'symlink':
                src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                os.symlink(src_relpath, tar_path)
            elif mode == 'copy':
                if osp.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path,dirs_exist_ok=True)
                else:
                    warnings.warn(f'Cannot copy file {src_path}.')
            else:
                raise ValueError(f'Invalid mode {mode}')


if __name__ == '__main__':
    print(get_require('./requirements.txt'))
    add_mim_extension()
    setup(
        name='edgelab',
        version=get_version(),
        description=
        'Seeed Studio EdgeLab is an open-source project focused on embedded AI. \
            We optimize the excellent algorithms from OpenMMLab for real-world scenarios and make implemention more user-friendly,\
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
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            'License :: OSI Approved :: MIT License',
            'Operating System :: Unix',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
        ],
        # install_requires=get_require('./requirements.txt'),
        include_package_data=True,
        python_requires='>=3.5',
        license='MIT',
        entry_points={
            'console_scripts': [
                'edgtrain=edgelab.tools.train:main',
                'edginfer=edgelab.tools.test:main',
                'edgenv=edgelab.tools.env_config:main',
                'edgexport=edgelab.tools.export:main',
                'edg2onnx=edgelab.tools.torch2onnx:main'
            ]
        },
    )
