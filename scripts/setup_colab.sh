#!/bin/bash


# configs
INSTALL_OPTIONAL=true
INSTALL_DOCS=false
CUDA_AVAILABLE="$(command -v nvidia-smi)"


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
if [ ! "${CUDA_AVAILABLE}" ]; then
    echo -en "${BLUE}Using CUDA${RST}\n"
    pip3 -r requirements/pytorch_cuda.txt && \
    pip3 -r requirements/base.txt
else
    echo -en "${BLUE}Using CPU${RST}\n"
    pip3 -r requirements/pytorch_cpu.txt && \
    pip3 -r requirements/base.txt
fi
if [ "$?" != 0 ]; then
    echo -en "Install base deps failed... ${RED}Exiting${RST}\n"
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
if [ "${INSTALL_OPTIONAL}" ]; then
    pip3 install -r requirements/inference.txt
fi


# install docs deps
if [ "${INSTALL_DOCS}" ]; then
    npm ci
fi


echo -en "Finished setup... ${GREEN}OK${RST}\n"
conda deactivate

exit 0
