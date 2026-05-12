ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Install library dependencies
WORKDIR /workspace

## Unset NVTE_CK_FUSED_ATTN_PATH from base image so TE's cmake uses prebuilt download
ENV NVTE_CK_FUSED_ATTN_PATH=""

# TE
RUN pip uninstall -y transformer-engine
ARG TE_COMMIT=261eda6ab09e603d816af78d31ba793f643f28fd
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

RUN cd TransformerEngine \
    && NVTE_PROJECT_BUILDING=1 NVTE_USE_ROCM=1 NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_ROCM_ARCH="gfx950;gfx942" GPU_ARCHS="gfx950;gfx942" \
    NVTE_FRAMEWORK='pytorch' \
    NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx950;gfx942" CU_NUM=304 \
    pip install --no-build-isolation -e . -v \
    && rm -rf /root/.cache

# AITER
ARG AITER_COMMIT=77455e3ecf4f0d28756afc452e914940c45b944b
RUN rm -rf /workspace/aiter \
    && git clone --recursive https://github.com/ROCm/aiter.git \
    && cd aiter && git checkout ${AITER_COMMIT}
ENV PYTHONPATH="/workspace/aiter:${PYTHONPATH}"
RUN echo "/workspace/aiter" > \
    $(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/aiter_workspace.pth

# Pre-compile AITER JIT modules
RUN pip install psutil
COPY precompile_aiter.py /tmp/precompile_aiter.py
RUN python3 /tmp/precompile_aiter.py

# Primus
ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
ARG PRIMUS_BRANCH=d53c428944c3d74c41c4c96fa2c1776d7e966538

RUN rm -rf /workspace/Primus && \
    git clone --recurse-submodules ${PRIMUS_REPO} /workspace/Primus && \
    cd /workspace/Primus && \
    git checkout ${PRIMUS_BRANCH} && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache

RUN cd /workspace/Primus/third_party/Megatron-LM && \
    pip install -e . --no-deps --no-build-isolation

WORKDIR /workspace/code

COPY . .

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#Install primus_mllog for MLPerf logging
COPY primus_mllog-0.1.21-py3-none-any.whl /tmp/primus_mllog-0.1.21-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.21-py3-none-any.whl && rm /tmp/primus_mllog-0.1.21-py3-none-any.whl
