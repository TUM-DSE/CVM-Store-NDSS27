FROM python:3.12-slim-trixie

ARG DEBIAN_FRONTEND=noninteractive
ARG SRC=FIXME
ARG DLIO_SRC=FIXME
# UV date
ARG DEPS_DATE=2026-08-25T00:00:00Z

ENV UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    PATH="/opt/mlcommons/storage:${PATH}" \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        build-essential \
        ca-certificates \
        curl \
        git \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-common \
        openssh-client \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/0.12.5/install.sh | sh

WORKDIR /opt/mlcommons

COPY ${SRC} /opt/mlcommons/storage
COPY ${DLIO_SRC} /opt/mlcommons/dlio

WORKDIR /opt/mlcommons/storage

RUN uv pip install --system --no-cache --no-sources --exclude-newer "${DEPS_DATE}" \
        -e /opt/mlcommons/dlio -e . && \
    uv pip check --system && \
    python3 -c "import dlio_benchmark, pydantic" && \
    mlpstorage --help >/dev/null

ENTRYPOINT ["mlpstorage"]
