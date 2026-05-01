ARG BASE_IMAGE=rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN pip install pybind11
RUN pip install ninja
RUN pip install packaging
RUN /usr/bin/python3 -m pip install pyYAML

# Install library dependencies
WORKDIR /workspace/deps

# Megatron-core (feature/mxfp4-recipe: includes MXFP4 recipe + use_hadamard support)
ARG MEGATRON_COMMIT=f312f7db1ec26b30831aae0fac023c6895a4ac1e
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
RUN cd megatron_lm && git checkout ${MEGATRON_COMMIT}   \
    && pip install -e . && cd megatron/core/datasets && make

# Use TE fused cross-entropy (Parallel CE from Transformer Engine instead of MCore native)
RUN cd megatron_lm \
    && sed -i "s/cross_entropy_fusion_impl: str = 'native'/cross_entropy_fusion_impl: str = 'te'/" \
       megatron/core/model_parallel_config.py

ENV PYTHONPATH "${PYTHONPATH}:/workspace/deps/megatron_lm"

# NeMo - keep ROCm torch, use constraints
RUN pip freeze | grep "torch" > /tmp/torch-constraints.txt || echo "torch" > /tmp/torch-constraints.txt && \
    cat /tmp/torch-constraints.txt

# NeMo v2.6.0 (llama2_sft: 2025-12-03)
RUN git clone https://github.com/NVIDIA/NeMo nemo \
    && cd nemo && git checkout v2.6.0 \
    && sed -i '12d' requirements/requirements_nlp.txt
RUN cd /workspace/deps/nemo \
    && PIP_CONSTRAINT=/tmp/torch-constraints.txt pip install --no-build-isolation -e ".[nlp,multimodal,llm]"

# NeMo-Run
RUN pip install git+https://github.com/NVIDIA/NeMo-Run.git@v0.4.0

# Python deps
# Important: install after NeMo so transformers pin is overridden to the version we need
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python -m ensurepip || apt-get install -y python3-venv \
&& pip install cmake==3.28.3

# Transformer Engine 2.12
ARG TE_COMMIT=261eda6ab09e603d816af78d31ba793f643f28fd
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

RUN cd TransformerEngine \
    && NVTE_PROJECT_BUILDING=1 NVTE_USE_ROCM=1 NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_ROCM_ARCH="gfx950" GPU_ARCHS="gfx950" \
    NVTE_FRAMEWORK='pytorch' \
    NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx950" CU_NUM=304 \
    pip install --no-build-isolation -e . -v

# AITER — required for MXFP4 FP4 GEMM kernels
# Pinned to known-good pybind commit; unpinned HEAD regresses MXFP4 throughput by ~4%
ARG AITER_COMMIT=77455e3ecf4f0d28756afc452e914940c45b944b
RUN git clone --recursive https://github.com/ROCm/aiter.git \
    && cd aiter && git checkout ${AITER_COMMIT}
ENV PYTHONPATH="/workspace/deps/aiter:${PYTHONPATH}"

# Pre-compile AITER JIT modules at build time (avoids ~20s+ runtime penalty)
COPY precompile_aiter.py /tmp/precompile_aiter.py
RUN pip install psutil && python3 /tmp/precompile_aiter.py

# Uninstall bitsandbytes to remove NVIDIA related warnings
RUN pip uninstall -y bitsandbytes mlperf-logging megatron-energon mlperf-common

RUN pip install megatron-energon==7.1.0
RUN pip install git+https://github.com/mlcommons/logging.git@6.0.0-rc1
RUN pip install git+https://github.com/NVIDIA/mlperf-common.git@b86d175a05849d650a8ff69c1e2c37b9f4e61d51

WORKDIR /workspace/code

# Copy the current state of the code inside the image
COPY . .
