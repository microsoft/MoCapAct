FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew2.0 \
    libosmesa6-dev \
    libglfw3 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    net-tools \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    xvfb \
    pv \
    rsync \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV DISPLAY :0

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

# CMU Mocap Data
RUN mkdir /opt/cmu_mocap_data
COPY ./cmu_2020_dfe3e9e0.h5 /opt/cmu_mocap_data
ENV CMU_MOCAP_DIR /opt/cmu_mocap_data

# Azcopy
COPY ./azcopy /usr/bin/

RUN rm /usr/bin/python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip

# Pytorch 1.8.1
RUN pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Other deps
RUN pip install dm_control
RUN pip install pytorch-lightning
RUN pip install ml_collections
RUN pip install git+https://github.com/DLR-RM/stable-baselines3.git@master

# This overcomes a bug in AzureML in which conda reports: ModuleNotFoundError: No module named 'ruamel'
RUN rm -fr /opt/miniconda/lib/python3.7/site-packages/ruamel_yaml*
RUN pip install ruamel_yaml

# Create a virtual display
# RUN xvfb-run -a -s "-screen 0 1400x900x24" bash
