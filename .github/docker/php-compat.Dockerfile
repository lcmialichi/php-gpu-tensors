ARG PHP_VERSION=8.5
ARG PHP_MODE=cli
FROM php:${PHP_VERSION}-${PHP_MODE}-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf build-essential libtool pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN php --version && phpize --version
