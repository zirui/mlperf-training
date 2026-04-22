ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Install library dependencies
WORKDIR /workspace/deps

# Primus-Turbo
ARG PRIMUS_TURBO_COMMIT=c79e0f359a0c0952c634f3b48312cd6341b5d843
RUN git clone --recursive https://github.com/AMD-AGI/Primus-Turbo.git && \
    cd Primus-Turbo && \
    git checkout ${PRIMUS_TURBO_COMMIT} && \
    git submodule update --init --recursive

RUN pip uninstall amd-aiter aiter -y 2>/dev/null || true && \
    cd /workspace/deps/Primus-Turbo && \
    pip3 install -r requirements.txt && \
    PYTORCH_ROCM_ARCH="gfx950" GPU_ARCHS="gfx950" pip3 install --no-build-isolation . -v

# Transformer Engine
RUN pip uninstall transformer-engine transformer_engine -y 2>/dev/null || true

ARG TE_COMMIT=575447c5b424489b58870110b63bbacc4dff5c21
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

COPY patches/aiter_fwd_v3_silent_fallback.patch /tmp/aiter_fwd_v3_silent_fallback.patch
RUN cd TransformerEngine && \
    git apply /tmp/aiter_fwd_v3_silent_fallback.patch && \
    NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_CK_FUSED_ATTN_PATH="" \
    NVTE_BUILD_MAX_JOBS=128 NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH=gfx950 \
    NVTE_USE_HIPBLASLT=1 PYTORCH_ROCM_ARCH="gfx950" CU_NUM=304 \
    pip install --no-build-isolation -e . -v

# Patches 
ADD patches /workspace/deps/patches

# Transformer Engine patches
RUN cd TransformerEngine \
    && git apply /workspace/deps/patches/te_fmha_backend_selection.patch \
    && git apply /workspace/deps/patches/aiter_fused_rope.patch

# Megatron-LM patches
RUN cd /workspace/Primus/third_party/Megatron-LM && \
    git apply /workspace/deps/patches/megatron_validation_consumed_samples.patch && \
    git apply /workspace/deps/patches/megatron_cuda_sync_model_init.patch && \
    git apply /workspace/deps/patches/megatron_te_general_gemm_workspace_compat.patch && \
    git apply /workspace/deps/patches/megatron_te_bshd_layout.patch && \
    git apply /workspace/deps/patches/megatron_moe_skip_identity_sort.patch && \
    pip install -e . --no-deps --no-build-isolation

# Primus patches
# 1. primus_turbo_te_compat: fix pre-existing dist_group_type import path and
#    FP8GlobalStateManager.fp8_autocast_{enter,exit} -> .autocast_{enter,exit}
#    rename so turbo-related Primus patches actually load against the shipped TE.
# 2. turbo_rms_norm: swap stock TE RMSNorm and fused TELayerNormColumnParallelLinear
#    for Triton-backed PrimusTurbo equivalents at the Primus patch-runner level.
RUN cd /workspace/Primus && \
    git apply /workspace/deps/patches/primus_turbo_te_compat.patch && \
    git apply /workspace/deps/patches/turbo_rms_norm.patch

# Ship the Triton RMSNorm kernel alongside the patched primus_turbo module.
COPY patches/triton_rmsnorm.py \
     /workspace/Primus/primus/backends/megatron/core/extensions/triton_rmsnorm.py

WORKDIR /workspace/code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the current state of the code inside the image
COPY . .

COPY primus_mllog-0.1.10-py3-none-any.whl /tmp/primus_mllog-0.1.10-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.10-py3-none-any.whl && rm /tmp/primus_mllog-0.1.10-py3-none-any.whl

# Primus PR #670: update triton permutation (feat/te/update_permutation)
COPY patches/primus_te_permutation/permutation.py \
     /workspace/Primus/primus/backends/transformer_engine/pytorch/permutation.py
COPY patches/primus_te_permutation/triton/permutation.py \
     /workspace/Primus/primus/backends/transformer_engine/pytorch/triton/permutation.py
COPY patches/primus_te_permutation/common/triton/permutation.py \
     /workspace/Primus/primus/backends/transformer_engine/pytorch/common/triton/permutation.py
