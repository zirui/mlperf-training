ARG BASE_IMAGE=rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN pip install pybind11
RUN pip install ninja
RUN pip install packaging
RUN /usr/bin/python3 -m pip install pyYAML

# Install library dependencies
WORKDIR /workspace/deps

ADD patches /workspace/deps/patches

# Megatron-core (feature/mxfp4-recipe: includes MXFP4 recipe + use_hadamard support)
ARG MEGATRON_COMMIT=3c1b9cd495a7c61b0c0473dffb565fc5ce9eb194
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
RUN cd megatron_lm && git checkout ${MEGATRON_COMMIT}   \
    && pip install -e . && cd megatron/core/datasets && make

# TE-native SwiGLU integration (replaces Triton pointwise fallback)
RUN cd megatron_lm && git apply /workspace/deps/patches/fused_bias_swiglu.patch

ENV PYTHONPATH "${PYTHONPATH}:/workspace/deps/megatron_lm"

# NeMo - keep ROCm torch, use constraints
RUN pip freeze | grep "torch" > /tmp/torch-constraints.txt || echo "torch" > /tmp/torch-constraints.txt && \
    cat /tmp/torch-constraints.txt

RUN git clone https://github.com/NVIDIA/NeMo nemo \
    && cd nemo && git checkout v2.6.0 \
    && sed -i '12d' requirements/requirements_nlp.txt
RUN cd /workspace/deps/nemo \
    && PIP_CONSTRAINT=/tmp/torch-constraints.txt pip install --no-build-isolation -e ".[nlp,multimodal,llm]"

# NeMo-Run
RUN pip install git+https://github.com/NVIDIA/NeMo-Run.git@v0.4.0

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python -m ensurepip || apt-get install -y python3-venv \
&& pip install cmake==3.28.3

# Transformer Engine 2.12 MXFP4
ARG TE_COMMIT=7775ce5aa76665bbc2a890c58287e2ad11d78e9d
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

RUN cd TransformerEngine && git apply /workspace/deps/patches/cast_transpose_mxfp4_v2_te212.patch
RUN cd TransformerEngine && git apply /workspace/deps/patches/hadamard_fallback.patch

RUN cd TransformerEngine \
    && NVTE_PROJECT_BUILDING=1 NVTE_USE_ROCM=1 NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_ROCM_ARCH="gfx950" GPU_ARCHS="gfx950" \
    NVTE_FRAMEWORK='pytorch' \
    NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx950" CU_NUM=304 \
    pip install --no-build-isolation -e . -v

# AITER - Required for MXFP4 FP4 GEMM kernels
# Pinned to known-good commit for FP4 GEMM throughput (unpinned HEAD regresses MXFP4 by ~4%)
ARG AITER_COMMIT=77455e3ecf4f0d28756afc452e914940c45b944b
RUN git clone --recursive https://github.com/ROCm/aiter.git \
    && cd aiter && git checkout ${AITER_COMMIT}
ENV PYTHONPATH="/workspace/deps/aiter:${PYTHONPATH}"

# Pre-compile AITER JIT modules
RUN pip install psutil
COPY precompile_aiter.py /tmp/precompile_aiter.py
RUN python3 /tmp/precompile_aiter.py

# AITER fused RoPE: patch TE to dispatch to aiter.ops.rope when available
RUN cd TransformerEngine && git apply /workspace/deps/patches/rope.patch

# Uninstall bitsandbytes to remove NVIDIA related warnings
RUN pip uninstall -y bitsandbytes mlperf-logging megatron-energon mlperf-common

RUN pip install megatron-energon==7.1.0
RUN pip install git+https://github.com/mlcommons/logging.git@6.0.0-rc1
RUN pip install git+https://github.com/NVIDIA/mlperf-common.git@b86d175a05849d650a8ff69c1e2c37b9f4e61d51

WORKDIR /workspace/code

COPY . .
