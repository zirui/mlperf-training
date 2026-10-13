# syntax=docker/dockerfile:1.7
# FBGEMM-GPU has no public gfx950 / ROCm 7.2 wheel, so the fbgemm stage builds
# it from source.
ARG BASE_IMAGE=rocm/primus:v26.3@sha256:1a02b74a94d82131f3f119a94ccbed45bb5b4cd77a0616cb4e6a29e64a048482

FROM ${BASE_IMAGE} AS stack
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Replaces the base image's primus-mllog with the AMD-AGI/mlperf-common primus
# mllog. --no-deps keeps mlcommons/logging from requirements.txt.
ARG MLPERF_COMMON_REF=7771aeabc8b91402e95cb7a33f447d9d9941361f
RUN python3 -m pip uninstall -y primus-mllog 2>/dev/null || true
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    python3 -m pip install --no-deps --no-cache-dir \
      "git+https://github.com/AMD-AGI/mlperf-common.git@${MLPERF_COMMON_REF}"

# torch, torchvision and torchaudio by URL, over the base image's torch
RUN pip install --no-deps --force-reinstall --no-cache-dir \
    "torch @ https://download.pytorch.org/whl/rocm7.2/torch-2.12.0%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "torchvision @ https://download.pytorch.org/whl/rocm7.2/torchvision-0.27.0%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl" \
    "torchaudio @ https://download.pytorch.org/whl/rocm7.2/torchaudio-2.11.0%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl"

# --no-deps, or pip replaces the ROCm torch with a CUDA one
ARG TORCHREC_GIT=https://github.com/pytorch/torchrec.git
ARG TORCHREC_COMMIT=bf554808b185cc7c694525e34ae2b910311842cf
RUN pip install --no-deps --force-reinstall "torchrec @ git+${TORCHREC_GIT}@${TORCHREC_COMMIT}"

FROM stack AS fbgemm
ARG FBGEMM_GIT=https://github.com/pytorch/FBGEMM.git
ARG FBGEMM_COMMIT=10b775730212923f65f7b78f79b6a01d80cf3c29
ARG FBGEMM_VERSION_DATE=2026,6,2
ARG AMDGPU_TARGETS=gfx950
ARG MAX_JOBS=32
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*
RUN git clone "${FBGEMM_GIT}" /src/FBGEMM && cd /src/FBGEMM && \
    git checkout "${FBGEMM_COMMIT}" && git submodule update --init --recursive
# Nightly builds name themselves after the build date; pin it to 2026.6.2.
RUN cd /src/FBGEMM/fbgemm_gpu && \
    grep -q "today = date.today()" setup.py && \
    sed -i "s/today = date.today()/today = date(${FBGEMM_VERSION_DATE})/" setup.py && \
    grep -q "today = date(${FBGEMM_VERSION_DATE})" setup.py
RUN cd /src/FBGEMM/fbgemm_gpu && \
    grep -v -E '^(fairscale|torch|torchvision|torchaudio)([<>=!]|$)' requirements.txt > /tmp/fbgemm_req.txt && \
    pip install -r /tmp/fbgemm_req.txt && \
    python setup.py -j "${MAX_JOBS}" bdist_wheel \
        --build-target=default --build-variant=rocm \
        -DHIP_ROOT_DIR=/opt/rocm -DAMDGPU_TARGETS="${AMDGPU_TARGETS}" && \
    mkdir -p /wheels && cp dist/fbgemm_gpu_nightly_rocm-*.whl /wheels/

FROM stack
COPY --from=fbgemm /wheels/ /tmp/wheels/
RUN pip install --no-deps --force-reinstall /tmp/wheels/fbgemm_gpu_nightly_rocm-*.whl && \
    rm -rf /tmp/wheels

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=/workspace/code

WORKDIR /workspace/code

COPY . .
