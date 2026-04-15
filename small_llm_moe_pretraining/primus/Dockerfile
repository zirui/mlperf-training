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

ARG TE_COMMIT=2aa87bf4e30917b69ff1ca33e7f71c4bf735ce3a
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
    pip install -e . --no-deps --no-build-isolation

WORKDIR /workspace/code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the current state of the code inside the image
COPY . .

COPY primus_mllog-0.1.1-py3-none-any.whl /tmp/primus_mllog-0.1.1-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.1-py3-none-any.whl && rm /tmp/primus_mllog-0.1.1-py3-none-any.whl
