#!/bin/bash


# configs
INSTALL_OPTIONAL=true
INSTALL_DOCS=false
CUDA_AVAILABLE="$(command -v nvidia-smi)"
PYTHON_PATH="/usr/bin/python3.8"


# ansi colors
RED='\033[031m'
GREEN='\033[032m'
BLUE='\033[034m' 
RST='\033[m'


# check cuda
echo -en "Checking if CUDA available... "
if [ ! "${CUDA_AVAILABLE}" ]; then
    echo -en "${RED}Not found!${RST}\n"
    echo -en "Using CPU instead... ${BLUE}$(lscpu | sed -nr '/Model name/ s/.*:\s*(.*) @ .*/\1/p')${SYSTEM_ARCH}${RST}\n"
else
    echo -en "${GREEN}OK${RST}\n"
fi


# install base deps
echo -en "Installing base deps... "
if [ "${CUDA_AVAILABLE}" ]; then
    echo -en "${BLUE}Using CUDA${RST}\n"
    PYTHON_PATH -m pip install -r requirements/pytorch_cuda.txt && \
    PYTHON_PATH -m pip install -r requirements/base.txt
else
    echo -en "${BLUE}Using CPU${RST}\n"
    PYTHON_PATH -m pip install -r requirements/pytorch_cpu.txt && \
    PYTHON_PATH -m pip install -r requirements/base.txt
fi
if [ "$?" != 0 ]; then
    echo -en "Install base deps failed... ${RED}Exiting${RST}\n"
    exit 1
fi


# openmim install deps
echo -en "Installing OpenMIM deps... \n"
PYTHON_PATH -m mim install -r requirements/mmlab.txt && \
PYTHON_PATH -m mim install -e .
if [ "$?" != 0 ]; then
    echo -en "OpenMIM install deps failed... ${RED}Exiting${RST}\n"
    exit 1
fi


# install optional deps
if [ "${INSTALL_OPTIONAL}" ]; then
    PYTHON_PATH -m pip install -r requirements/inference.txt
fi


# install docs deps
if [ "${INSTALL_DOCS}" == true ]; then
    npm i
fi


echo -en "Finished setup... ${GREEN}OK${RST}\n"
exit 0
