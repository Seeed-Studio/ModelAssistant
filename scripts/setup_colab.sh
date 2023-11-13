#!/bin/bash


# configs
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
    echo -en "Please enable GPU Runtime${RST}\n"
    exit 1
else
    echo -en "${GREEN}OK${RST}\n"
fi

# limit torch version
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1

# install base deps
echo -en "Installing base deps... "
pip install -r requirements/base.txt -r requirements/inference.txt -r requirements/export.txt -r requirements/tests.txt

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

echo -en "Finished setup... ${GREEN}OK${RST}\n"
exit 0
