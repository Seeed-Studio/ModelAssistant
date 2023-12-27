# Start FROM PyTorch image https://hub.docker.com/r/pytorch/pytorch
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg build-essential

# Get SSCMA
RUN git clone https://github.com/Seeed-Studio/ModelAssistant -b main --depth 1

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel
RUN cd SSCMA && pip install --no-cache -r requirements/base.txt
RUN cd SSCMA && pip install --no-cache -r requirements/export.txt
RUN cd SSCMA && pip install --no-cache -r requirements/inference.txt
RUN cd SSCMA && mim install --no-cache -r requirements/mmlab.txt
RUN cd SSCMA && pip install .

WORKDIR /workspace

# Set environment variables
ENV OMP_NUM_THREADS=1

# Cleanup
RUN rm -rf SSCMA
ENV DEBIAN_FRONTEND teletype
