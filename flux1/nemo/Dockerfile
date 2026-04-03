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

# Megatron-core (llama2_sft: 2026-03-03, includes MXFP4 recipe + fused CE TE)
ARG MEGATRON_COMMIT=9a37db9be7276a74d587fd617536de9933d4ae55
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
RUN cd megatron_lm && git checkout ${MEGATRON_COMMIT}   \
    && pip install -e . && cd megatron/core/datasets && make

# TE-native SwiGLU integration (replaces Triton pointwise fallback)
RUN cd megatron_lm && git apply /workspace/deps/patches/te_native_swiglu.patch

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

# Transformer Engine (TE 2.8 + MXFP4, 2026-03-31)
ARG TE_COMMIT=6f406e55cd2ab7590d8e593c71350a77d2c930c2
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

# CUDAGuard fix for MXFP4 standalone quantization on multi-GPU
RUN cd TransformerEngine && git apply /workspace/deps/patches/mxfp4_hip_cudaguard.patch

ARG NVTE_AITER_PREBUILT_BASE_URL_DEFAULT=https://compute-artifactory.amd.com:5000/artifactory/rocm-generic-local/te-ci/aiter-prebuilts
RUN cd TransformerEngine \
    && PREBUILT_URL="" \
    && if curl -sS --connect-timeout 5 --max-time 10 -o /dev/null "${NVTE_AITER_PREBUILT_BASE_URL_DEFAULT}" 2>/dev/null; then \
         PREBUILT_URL="${NVTE_AITER_PREBUILT_BASE_URL_DEFAULT}"; \
         echo "AITER prebuilt server reachable, using prebuilts"; \
       else \
         echo "AITER prebuilt server unreachable, building from source"; \
       fi \
    && NVTE_PROJECT_BUILDING=1 NVTE_USE_ROCM=1 NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_ROCM_ARCH="gfx942;gfx950" GPU_ARCHS="gfx942;gfx950" \
    NVTE_FRAMEWORK='pytorch' \
    NVTE_AITER_PREBUILT_BASE_URL="${PREBUILT_URL}" \
    NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx942;gfx950" CU_NUM=304 \
    pip install --no-build-isolation -e . -v

# AITER v0.1.12 (2026-03-30) — standalone install for MXFP4 quantization
ARG AITER_BRANCH="v0.1.12"
RUN git clone https://github.com/ROCm/aiter.git \
    && cd aiter \
    && git checkout ${AITER_BRANCH} \
    && git submodule sync && git submodule update --init --recursive \
    && pip install --no-build-isolation -e .

# Symlink AITER HSA kernels into TransformerEngine for F4 GEMM kernel discovery at runtime
RUN mkdir -p /workspace/deps/TransformerEngine/transformer_engine/aiter/gfx950 \
    && ln -sf /workspace/deps/aiter/hsa/gfx950 \
    /workspace/deps/TransformerEngine/transformer_engine/aiter/gfx950/gfx950

# Pre-compile AITER JIT modules at build time (avoids ~20s+ runtime penalty)
RUN pip install psutil && \
    GPU_ARCHS="gfx942;gfx950" python3 -c "\
import os; os.environ['GPU_ARCHS'] = 'gfx942;gfx950'; \
from aiter.jit.core import get_args_of_build, build_module; \
[build_module(m, **{k: v for k, v in get_args_of_build(m).items() if k not in ('md_name', 'hip_clang_path')}) \
  for m in ('module_rope_general_fwd','module_rope_general_bwd','module_gemm_a4w4_asm')]; \
print('AITER RoPE+FP4 GEMM pre-compiled OK')"

# MXFP4 weight cache + lazy columnwise quant (LoRA: no wgrad for base weights)
# AITER fused RoPE: dispatch to aiter.ops.rope when available
RUN cd TransformerEngine && git apply /workspace/deps/patches/mxfp4_te_modules.patch \
    && git apply /workspace/deps/patches/aiter_fused_rope.patch

# Uninstall bitsandbytes to remove NVIDIA related warnings
RUN pip uninstall -y bitsandbytes mlperf-logging megatron-energon mlperf-common

RUN pip install megatron-energon==7.1.0
RUN pip install git+https://github.com/mlcommons/logging.git@34903058ee63d3a7c3bcc1a6ca04e2aa5ed6af9c
RUN pip install git+https://github.com/NVIDIA/mlperf-common.git@b86d175a05849d650a8ff69c1e2c37b9f4e61d51

# AINIC driver (Pensando ionic NIC support for multi-node RDMA, optional)
# Dockerfile is always present, so COPY never fails even if the tarball is missing.
COPY Dockerfile ainic_bundle*.tar.gz /tmp/ainic/
RUN if ls /tmp/ainic/ainic_bundle*.tar.gz 1>/dev/null 2>&1; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            jq dpkg-dev kmod xz-utils ethtool iproute2 \
            libibverbs-dev ibverbs-utils infiniband-diags && \
        rm -rf /var/lib/apt/lists/* && \
        cd /tmp/ainic && tar zxf ainic_bundle*.tar.gz && \
        cd ainic_bundle_* && tar zxf host_sw_pkg.tar.gz && \
        cd host_sw_pkg && ./install.sh --domain=user -y; \
    else \
        echo "AINIC bundle not found, skipping multi-node RDMA support"; \
    fi && rm -rf /tmp/ainic

WORKDIR /workspace/code

# Copy the current state of the code inside the image
COPY . .
