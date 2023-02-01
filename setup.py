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


if __name__ == '__main__':
    print(get_require('./requirements.txt'))
    setup(
        name='edgelab',
        version=get_version(),
        description=
        'Seeed Studio EdgeLab is an open-source project focused on embedded AI. \
            We optimize the excellent algorithms from OpenMMLab for real-world scenarios and make implemention more user-friendly,\
             achieving faster and more accurate inference on embedded devices.',
        long_description=readme(),
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
        install_requires=get_require('./requirements.txt'),
        include_package_data=True,
        python_requires='>=3.5',
        license='MIT',
    )
