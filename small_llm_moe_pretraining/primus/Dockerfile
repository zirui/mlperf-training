ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Install library dependencies
WORKDIR /workspace/deps

ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
# tip of mpo branch
ARG PRIMUS_REF=d6f5ca40f398ca316acbf9e682dd475383e440fc
RUN rm -rf /workspace/Primus && \
    git clone --recursive "${PRIMUS_REPO}" /workspace/Primus && \
    cd /workspace/Primus && \
    git checkout "${PRIMUS_REF}" && \
    git submodule sync --recursive && \
    git submodule update --init --recursive


ARG PRIMUS_TURBO_REPO=https://github.com/AMD-AGI/Primus-Turbo.git
# tip of mpo branch
ARG PRIMUS_TURBO_REF=5a3fc741dfa2e1feb33f00c261e446fb5ae5e44c
RUN git clone --recursive "${PRIMUS_TURBO_REPO}" /workspace/deps/Primus-Turbo && \
    cd /workspace/deps/Primus-Turbo && \
    git checkout "${PRIMUS_TURBO_REF}" && \
    git submodule sync --recursive && \
    git submodule update --init --recursive

RUN pip uninstall amd-aiter aiter -y 2>/dev/null || true && \
    cd /workspace/deps/Primus-Turbo && \
    pip3 install -r requirements.txt && \
    PYTORCH_ROCM_ARCH="gfx950" GPU_ARCHS="gfx950" pip3 install --no-build-isolation . -v

RUN pip uninstall transformer-engine transformer_engine -y 2>/dev/null || true

ARG TE_COMMIT=bcab0d308bc6d992089396b8fd3c23d109c30b2e
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

COPY patches/aiter_fwd_v3_silent_fallback.patch /tmp/aiter_fwd_v3_silent_fallback.patch
COPY patches/te_fmha_backend_selection.patch   /tmp/te_fmha_backend_selection.patch
COPY patches/aiter_fused_rope.patch            /tmp/aiter_fused_rope.patch

RUN cd TransformerEngine && \
    git apply /tmp/aiter_fwd_v3_silent_fallback.patch && \
    git apply /tmp/te_fmha_backend_selection.patch && \
    git apply /tmp/aiter_fused_rope.patch && \
    NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_CK_FUSED_ATTN_PATH="" \
    NVTE_BUILD_MAX_JOBS=128 NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH=gfx950 \
    NVTE_USE_HIPBLASLT=1 PYTORCH_ROCM_ARCH="gfx950" CU_NUM=304 \
    pip install --no-build-isolation -e . -v

RUN cd /workspace/Primus/third_party/Megatron-LM && \
    for patch in /workspace/Primus/primus/backends/megatron/patches/megatron_lm_source/*.patch; do \
        echo "Applying $(basename "$patch")" && \
        git apply "$patch"; \
    done && \
    pip install -e . --no-deps --no-build-isolation

WORKDIR /workspace/code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the current state of the code inside the image
COPY . .

COPY primus_mllog-0.1.15-py3-none-any.whl /tmp/primus_mllog-0.1.15-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.15-py3-none-any.whl && rm /tmp/primus_mllog-0.1.15-py3-none-any.whl

# Fused rmsnor and swiglu no cat
COPY patches/primus_fused_rms_swiglu_nocat/fused_residual_rmsnorm.py \
     /workspace/Primus/primus/backends/megatron/core/extensions/fused_residual_rmsnorm.py
COPY patches/primus_fused_rms_swiglu_nocat/moe_swigly_nocat.py \
     /workspace/Primus/primus/backends/megatron/core/extensions/moe_swiglu_nocat.py
COPY patches/primus_fused_rms_swiglu_nocat/triton_rmsnorm.py \
     /workspace/Primus/primus/backends/megatron/core/extensions/triton_rmsnorm.py
COPY patches/primus_fused_rms_swiglu_nocat/triton_swiglu.py \
     /workspace/Primus/primus/backends/megatron/core/extensions/triton_swiglu.py
COPY patches/primus_fused_rms_swiglu_nocat/swigly_nocat_patches.py \
     /workspace/Primus/primus/backends/megatron/patches/moe_patches/swiglu_nocat_patches.py
COPY patches/primus_fused_rms_swiglu_nocat/fused_residual_norm_patches.py \
     /workspace/Primus/primus/backends/megatron/patches/turbo/swiglu_nocat_patches.py