ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG PHP_VERSION=8.1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    php${PHP_VERSION}-cli \
    php${PHP_VERSION}-dev \
    build-essential \
    autoconf \
    pkg-config \
    libtool && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/ext

CMD ["php", "-a"]