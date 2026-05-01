ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

WORKDIR /workspace/

## AITER: clone + setup.py develop (builds Python C extensions / JIT modules)
## MHA kernel compilation (compile.py) is skipped — prebuilts are downloaded by TE via artifactory.
## The mha.py sed patch still applies: it changes a *runtime* default, not a compile-time setting.
ARG AITER_REPO=https://github.com/rocm/aiter.git
ARG AITER_COMMIT=d32b0cb62ecc32bb1a858e8437d58eb9b3856af6
RUN rm -rf /workspace/aiter \
    && git clone --recursive ${AITER_REPO} \
    && cd aiter \
    && git checkout ${AITER_COMMIT} \
    && git submodule update --init --recursive \
    && find aiter/ops/mha.py -type f -print0 | xargs -0 sed -i 's/is_v3_atomic_fp32: Optional\[bool\] = True,/is_v3_atomic_fp32: Optional\[bool\] = False,/g' \
    && python3 -c "import json; p='aiter/jit/optCompilerConfig.json'; c=json.load(open(p)); c['module_aiter_enum']['torch_exclude']='False'; json.dump(c,open(p,'w'),indent=4)" \
    && rm -rf aiter/jit/build \
    && python3 setup.py develop \
    && cd .. \
    && rm -rf /root/.cache

# Pre-compile AITER JIT modules at build time (avoids ~20s+ runtime penalty)
COPY precompile_aiter.py /tmp/precompile_aiter.py
RUN pip install psutil && python3 /tmp/precompile_aiter.py

## Unset NVTE_CK_FUSED_ATTN_PATH from base image so TE's cmake uses prebuilt download
ENV NVTE_CK_FUSED_ATTN_PATH=""

## Using TE branch origin/feature/mxfp4-recipe-212
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

RUN rm -rf Primus/
# Primus
ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
ARG PRIMUS_BRANCH=d53c428944c3d74c41c4c96fa2c1776d7e966538

RUN git clone --recurse-submodules ${PRIMUS_REPO} \
    && cd Primus \
    && git checkout ${PRIMUS_BRANCH} \
    && git submodule update --init --recursive \
    && pip install -r requirements.txt \
    && cd .. \
    && rm -rf /root/.cache

# Install Megatron-LM from bundled Primus
RUN cd /workspace/Primus/third_party/Megatron-LM && \
    pip install -e . --no-deps --no-build-isolation

WORKDIR /workspace/code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

COPY primus_mllog-0.1.17-py3-none-any.whl /tmp/primus_mllog-0.1.17-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.17-py3-none-any.whl && rm /tmp/primus_mllog-0.1.17-py3-none-any.whl
