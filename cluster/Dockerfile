FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV LANG=C.UTF-8

# Needed by nvidia-container-runtime, if used
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    cuda-command-line-tools-12-2 \
    cuda-nvcc-12-2 \
    sudo \
    git \
    python3.11 \
    python3.11-venv \
    python3.11-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create the user
RUN echo $USER_GID $USER_UID \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


RUN mkdir -p /workdir
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH=$VIRTUAL_ENV/bin:$PATH

# Install package dependencies
COPY pyproject.toml /workdir
COPY README.md /workdir
WORKDIR /workdir
RUN python3.11 -m pip install .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
USER $USERNAME