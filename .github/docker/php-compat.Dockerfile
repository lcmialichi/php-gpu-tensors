ARG PHP_VERSION=8.5
ARG PHP_MODE=cli
FROM php:${PHP_VERSION}-${PHP_MODE}-bookworm AS php
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf build-essential libargon2-1 libcurl4t64 libonig5 \
    libreadline8t64 libsodium23 libsqlite3-0 libssl3t64 libtool \
    libxml2 pkg-config zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=php /usr/local /usr/local
COPY --from=php /usr/lib/x86_64-linux-gnu/libicu*.so.72* /usr/local/lib/
RUN ldconfig && php --version
