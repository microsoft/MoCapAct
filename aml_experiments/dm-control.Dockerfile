FROM mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    sudo \
    cpio \
    git \
    make \
    cmake \
    swig \
    libz-dev \
    unzip \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    zlib1g-dev \
    libxrandr2 \
    libxinerama-dev \
    libxi6 \
    libxcursor-dev \
    libosmesa6-dev \
    libglfw3 \
    libglfw3-dev \
    software-properties-common \
    net-tools \
    lsb-release \
    ack-grep \
    patchelf \
    wget \
    xpra \
    xserver-xorg-dev \
    xvfb \
    gnupg2 \
    bzip2 \
    libx11-6 \
    tmux \
    htop \
    gcc \    
    python-opengl \
    x11-xserver-utils \
    ffmpeg \
    mesa-utils \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a Python 3.7 environment
RUN conda install conda-build \
    && conda create -y --name dmcontrol python=3.7 pip=20.2.4 \
    && conda clean -ya
ENV CONDA_DEFAULT_ENV=dmcontrol

#    mesa-utils \
ENV DISPLAY :0
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /opt/ \
    && rm mujoco.zip

RUN echo "MuJoCo Pro Individual license activation key, number 7777, type 6.\n\nIssued to Everyone.\n\nExpires October 18, 2031.\n\nDo not modify this file. Its entire content, including the\nplain text section, is used by the activation manager.\n\n9aaedeefb37011a8a52361c736643665c7f60e796ff8ff70bb3f7a1d78e9a605\n0453a3c853e4aa416e712d7e80cf799c6314ee5480ec6bd0f1ab51d1bb3c768f\n8c06e7e572f411ecb25c3d6ef82cc20b00f672db88e6001b3dfdd3ab79e6c480\n185d681811cfdaff640fb63295e391b05374edba90dd54cc1e162a9d99b82a8b\nea3e87f2c67d08006c53daac2e563269cdb286838b168a2071c48c29fedfbea2\n5effe96fe3cb05e85fb8af2d3851f385618ef8cdac42876831f095e052bd18c9\n5dce57ff9c83670aad77e5a1f41444bec45e30e4e827f7bf9799b29f2c934e23\ndcf6d3c3ee9c8dd2ed057317100cd21b4abbbf652d02bf72c3d322e0c55dcc24\n" >> /opt/mujoco200_linux/mjkey.txt

ENV LD_LIBRARY_PATH /opt/mujoco200_linux/bin:${LD_LIBRARY_PATH}
ENV MUJOCO_PATH /opt/mujoco200_linux
ENV MJKEY_PATH /opt/mujoco200_linux/mjkey.txt
ENV MJLIB_PATH /opt/mujoco200_linux/bin/libmujoco200.so

# This is a hack required to make mujocopy to compile in gpu mode
# RUN mkdir -p /usr/lib/nvidia-000
# ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-000

# dm_control requirements
RUN pip install 'absl-py==0.9.0' \
                'dm-env==1.2' \
                'dm-tree' \
                'future==0.18.2' \
                'glfw==1.11.0' \
                'h5py==3.1.0' \
                'labmaze==1.0.3' \
                'lxml==4.6.3' \
                'mock==3.0.5' \
                'nose==1.3.7' \
                'nose-xunitmp==0.4.1' \
                'numpy==1.19.5' \
                'Pillow==8.3.2' \
                'protobuf==3.15.6' \
                'pyopengl==3.1.5' \
                'pyparsing==2.4.6' \
                'requests==2.24.0' \
                'scipy==1.5.4' \
                'setuptools!=50.0.0' \
                'tqdm==4.47.0' \
                'gym' \
                'torch' \
                'stable-baselines3[extra]' \
                'pytorch-lightning' \
                'ml_collections' \
                'azureml-core' \
                'azureml-defaults' \
                'azureml-mlflow' \
                'azureml-telemetry' \
                'onnxruntime-gpu>=1.7,<1.8'

# Pytorch 1.7.1
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Various common packages
RUN pip install matplotlib

# This overcomes a bug in AzureML in which conda reports: ModuleNotFoundError: No module named 'ruamel'
RUN rm -fr /opt/miniconda/lib/python3.7/site-packages/ruamel_yaml*
RUN pip install ruamel_yaml

# Create a virtual display
# RUN xvfb-run -a -s "-screen 0 1400x900x24" bash