#!/bin/bash


# configs
INSTALL_OPTIONAL=false
CONDA_AVAILABLE="$(command -v conda)"
CUDA_AVAILABLE="$(command -v nvidia-smi)"
SYSTEM_ARCH="$(dpkg --print-architecture)"


# ansi colors
RED='\033[031m'
GREEN='\033[032m'
BLUE='\033[034m' 
RST='\033[m'


# check conda
echo -en "Checking if conda installed... "
if [ ! "${CONDA_AVAILABLE}" ]; then
    echo -en "${RED}Not found!${RST}\n"
    echo -en "Trying install miniconda... ${BLUE}Arch: ${SYSTEM_ARCH}${RST}\n"
    MINICONDA_URL=""
    case "${SYSTEM_ARCH}" in
        "amd64")
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        ;;
        "aarch64")
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        ;;
        "arm64")
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        ;;
        "ppc64le")
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-ppc64le.sh"
        ;;
        "s390x")
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-s390x.sh"
        ;;
        *)
            echo -en "Conda not supported on your architecture... ${RED}Exiting!${RST}\n"
            exit 1
        ;;
    esac 

    wget "${MINICONDA_URL}" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh
    . ~/.bashrc

    if [ "$?" != 0 ] && [ "$(command -v conda)" ]; then
        echo -en "Conda install failed... ${RED}Exiting!${RST}\n"
        exit 1
    fi
    echo -en "Conda installed... ${GREEN}Path: $(which conda)${RST}\n"
else
    echo -en "${GREEN}Path: $(which conda)${RST}\n"
fi


# check cuda
echo -en "Checking if CUDA available... "
if [ ! "${CUDA_AVAILABLE}" ]; then
    echo -en "${RED}Not found!${RST}\n"
    echo -en "Using CPU instead... ${BLUE}$(lscpu | sed -nr '/Model name/ s/.*:\s*(.*) @ .*/\1/p')${SYSTEM_ARCH}${RST}\n"
else
    echo -en "${GREEN}OK!${RST}\n"
fi


# create conda env and install deps
echo -en "Creating conda env and installing base deps... \n"
if [ ! "${CUDA_AVAILABLE}" ]; then
    conda env create -n edgelab -f ./conda_cpu.yml
else
    conda env create -n edgelab -f ./conda_cuda.yml
fi
if [ "$?" != 0 ]; then
    echo -en "Conda create env failed... ${RED}Exiting!${RST}\n"
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate edgelab
if [ "$?" != 0 ]; then
    echo -en "Conda active env failed... ${RED}Exiting!${RST}\n"
    exit 1
fi


# openmim install deps
echo -en "Installing OpenMIM deps... \n"
mim install -r ./requirements/mmlab.txt && mim install -e .
if [ "$?" != 0 ]; then
    echo -en "OpenMIM install deps failed... ${RED}Exiting!${RST}\n"
    exit 1
fi


# install optionalm deps
if [ "${INSTALL_OPTIONAL}" ]; then
    # audio deps
    pip3 install -r requirements/audio.txt

    # inference deps
    pip3 install -r requirements/inference.txt

    # docs deps
    pip3 install -r requirements/docs.txt
fi

echo -en "Finished setup... ${GREEN}Exiting${RST}\n"
conda deactivate

exit 0
