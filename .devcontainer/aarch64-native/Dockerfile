# syntax=docker/dockerfile:1

ARG UBUNTU_RELEASE=noble


FROM ubuntu:${UBUNTU_RELEASE} AS benchmark-builder

ARG DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  ; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install --no-install-recommends --upgrade --yes \
      ca-certificates \
      cmake \
      curl \
      g++ \
      make

ARG BENCHMARK_VERSION=1.9.4

RUN --mount=type=tmpfs,target=/tmp \
    cd /tmp \
 && curl -L "https://github.com/google/benchmark/archive/refs/tags/v${BENCHMARK_VERSION}.tar.gz" -o benchmark.tar.gz \
 && tar xzf benchmark.tar.gz \
 && cd "benchmark-${BENCHMARK_VERSION}/" \
 && cmake . \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/benchmark \
      -D BENCHMARK_ENABLE_TESTING=OFF \
      -D BENCHMARK_INSTALL_DOCS=OFF \
 && cmake --build . --target install


FROM ubuntu:${UBUNTU_RELEASE}

ARG DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/apt.conf.d/docker-clean \
  ; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
 && apt-get install --no-install-recommends --upgrade --yes \
      clang \
      clang-format \
      cmake \
      cmake-format \
      g++ \
      git \
      less \
      libboost-dev \
      libclang-rt-dev \
      libgtest-dev \
      libnuma-dev \
      libomp-dev \
      make \
      ninja-build \
      qemu-user

COPY --from=benchmark-builder /opt/benchmark /opt/benchmark
