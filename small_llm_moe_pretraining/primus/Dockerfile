ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Install library dependencies
WORKDIR /workspace/deps

ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
# tip of mpo branch
ARG PRIMUS_REF=724c01fac52a4952c7a3a6021e5082852a2b3ff7
RUN rm -rf /workspace/Primus && \
    git clone --recursive "${PRIMUS_REPO}" /workspace/Primus && \
    cd /workspace/Primus && \
    git checkout "${PRIMUS_REF}" && \
    git submodule sync --recursive && \
    git submodule update --init --recursive


ARG PRIMUS_TURBO_REPO=https://github.com/AMD-AGI/Primus-Turbo.git
# tip of mpo branch
ARG PRIMUS_TURBO_REF=a4897f7bd6b00b2e95dd43c7cbc6fa9bcce3a68c
RUN git clone --recursive "${PRIMUS_TURBO_REPO}" /workspace/deps/Primus-Turbo && \
    cd /workspace/deps/Primus-Turbo && \
    git checkout "${PRIMUS_TURBO_REF}" && \
    git submodule sync --recursive && \
    git submodule update --init --recursive

RUN pip uninstall amd-aiter aiter -y 2>/dev/null || true && \
    cd /workspace/deps/Primus-Turbo && \
    pip3 install -r requirements.txt && \
    PYTORCH_ROCM_ARCH="gfx950;gfx942" GPU_ARCHS="gfx950;gfx942" pip3 install --no-build-isolation . -v

RUN pip uninstall transformer-engine transformer_engine -y 2>/dev/null || true

ARG TE_COMMIT=261eda6ab09e603d816af78d31ba793f643f28fd
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

COPY patches/aiter_fwd_v3_silent_fallback.patch /tmp/aiter_fwd_v3_silent_fallback.patch

RUN cd TransformerEngine && \
    git apply /tmp/aiter_fwd_v3_silent_fallback.patch && \
    NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_CK_FUSED_ATTN_PATH="" \
    NVTE_BUILD_MAX_JOBS=128 NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH="gfx950;gfx942" \
    NVTE_USE_HIPBLASLT=1 PYTORCH_ROCM_ARCH="gfx950;gfx942" CU_NUM=304 \
    pip install --no-build-isolation -e . -v

RUN cd /workspace/Primus/third_party/Megatron-LM && \
    for patch in /workspace/Primus/primus/backends/megatron/patches/megatron_lm_source/*.patch; do \
        echo "Applying $(basename "$patch")" && \
        git apply "$patch"; \
    done && \
    pip install -e . --no-deps --no-build-isolation

WORKDIR /workspace/code

# Copy the current state of the code inside the image
COPY . .

COPY primus_mllog-0.1.17-py3-none-any.whl /tmp/primus_mllog-0.1.17-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.17-py3-none-any.whl && rm /tmp/primus_mllog-0.1.17-py3-none-any.whl
