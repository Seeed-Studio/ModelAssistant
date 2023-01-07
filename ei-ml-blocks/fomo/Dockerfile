# syntax = docker/dockerfile:experimental
ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.3
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}.1-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA
ARG CUDNN=8.2.1.32-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=8.0.0-1
ARG LIBNVINFER_MAJOR_VERSION=8
# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Avoid confirmation dialogs
ENV DEBIAN_FRONTEND=noninteractive
# Makes Poetry behave more like npm, with deps installed inside a .venv folder
# See https://python-poetry.org/docs/configuration/#virtualenvsin-project
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# CUDA drivers
# SHELL ["/bin/bash", "-c"]
# COPY ./install_cuda.sh ./install_cuda.sh
# RUN ./install_cuda.sh && \
#     rm install_cuda.sh

# System dependencies
RUN apt update && apt install -y wget git python3 python3-pip zip jq

# Edgelab environment configuration

RUN pip3 install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0

RUN git clone https://github.com/Seeed-Studio/Edgelab --depth 1

RUN cd Edgelab && \
    pip3 install -r requirements/requirements.txt

RUN pip3 install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html

WORKDIR /scripts

# Copy the normal files (e.g. run.sh and the extract_dataset scripts in)
COPY . ./

ENTRYPOINT ["/bin/bash", "run.sh"]
