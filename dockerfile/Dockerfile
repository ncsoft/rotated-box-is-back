# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG UBUNTU_VERSION=16.04

ARG CUDA=10.0
FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA=10.0
ARG CUDADASH=10-0
ARG CUDNN=7.4.1.5-1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDADASH} \
        cuda-cublas-${CUDADASH} \
        cuda-cufft-${CUDADASH} \
        cuda-curand-${CUDADASH} \
        cuda-cusolver-${CUDADASH} \
        cuda-cusparse-${CUDADASH} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        wget

RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda${CUDA} \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

ARG PYTHON=python3.7

ENV LANG C.UTF-8

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y ${PYTHON}-dev

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN ${PYTHON} get-pip.py
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3
RUN ln -sf /usr/local/bin/pip /usr/local/bin/pip3

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

ARG TF_PACKAGE=tensorflow-gpu==1.14.0
RUN pip3 install --upgrade ${TF_PACKAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        nano \
        libpng-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libomp-dev

RUN pip3 install matplotlib pillow tqdm opencv-python==3.4.2.17 gast==0.2.2