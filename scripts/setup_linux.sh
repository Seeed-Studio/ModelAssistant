#!/bin/bash


# configs
INSTALL_OPTIONAL=true
INSTALL_DOCS=false
CONDA_AVAILABLE="$(command -v conda)"
CUDA_AVAILABLE="$(command -v nvidia-smi)"


# ansi colors
RED='\033[031m'
GREEN='\033[032m'
BLUE='\033[034m'
RST='\033[m'


# check conda
echo -en "Checking if conda installed... "
if [ ! "${CONDA_AVAILABLE}" ]; then
    echo -en "${RED}Not found${RST}\n"
    exit 1
else
    echo -en "${GREEN}Path: $(which conda)${RST}\n"
fi


# check cuda
echo -en "Checking if CUDA available... "
if [ ! "${CUDA_AVAILABLE}" ]; then
    echo -en "${RED}Not found!${RST}\n"
    echo -en "Using CPU instead... ${BLUE}$(lscpu | sed -nr '/Model name/ s/.*:\s*(.*) @ .*/\1/p')${SYSTEM_ARCH}${RST}\n"
else
    echo -en "${GREEN}OK${RST}\n"
fi


# create conda env and install deps
echo -en "Creating conda env and installing base deps... "
if [ "${CUDA_AVAILABLE}" ]; then
    echo -en "${BLUE}Using CUDA${RST}\n"
    conda env create -n sscma -f environment_cuda.yml
else
    echo -en "${BLUE}Using CPU${RST}\n"
    conda env create -n sscma -f environment.yml
fi
if [ "$?" != 0 ]; then
    echo -en "Conda create env failed... ${RED}Exiting${RST}\n"
    exit 1
fi

eval "$(conda shell.bash hook)" && \
conda activate sscma
if [ "$?" != 0 ]; then
    echo -en "Conda active env failed... ${RED}Exiting${RST}\n"
    exit 1
fi


# openmim install deps
echo -en "Installing OpenMIM deps... \n"
mim install -r requirements/mmlab.txt && \
mim install -e .
if [ "$?" != 0 ]; then
    echo -en "OpenMIM install deps failed... ${RED}Exiting${RST}\n"
    exit 1
fi


# install optional deps
if [ "${INSTALL_OPTIONAL}" == true ]; then
    pip3 install -r requirements/inference.txt -r requirements/export.txt -r requirements/tests.txt
    pre-commit install
fi


# install docs deps
if [ "${INSTALL_DOCS}" == true ]; then
    npm ci
fi


echo -en "Finished setup... ${GREEN}OK${RST}\n"
conda deactivate

exit 0
