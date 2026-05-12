ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Install library dependencies
WORKDIR /workspace/deps

ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
ARG PRIMUS_REF=8195b3e30f44b935511d5752176b1524dabeeb2c
RUN rm -rf /workspace/Primus && \
    git clone --recursive "${PRIMUS_REPO}" /workspace/Primus && \
    cd /workspace/Primus && \
    git fetch origin "${PRIMUS_REF}" && \
    git checkout "${PRIMUS_REF}" && \
    git submodule sync --recursive && \
    git submodule update --init --recursive


ARG PRIMUS_TURBO_REPO=https://github.com/AMD-AGI/Primus-Turbo.git
ARG PRIMUS_TURBO_REF=d372005bf487380b517044c8845031f4867d4389
RUN git clone --recursive "${PRIMUS_TURBO_REPO}" /workspace/deps/Primus-Turbo && \
    cd /workspace/deps/Primus-Turbo && \
    git checkout "${PRIMUS_TURBO_REF}" && \
    git submodule sync --recursive && \
    git submodule update --init --recursive

RUN pip uninstall amd-aiter aiter -y 2>/dev/null || true && \
    cd /opt && git clone -b release/3.7.x https://github.com/triton-lang/triton.git && \
    cd triton && pip3 install ninja cmake && \
    MAX_JOBS=8 pip3 install --no-build-isolation -v . && rm -rf /opt/triton && \
    cd /workspace/deps/Primus-Turbo && \
    pip3 install -r requirements.txt && \
    PYTORCH_ROCM_ARCH="gfx950;gfx942" GPU_ARCHS="gfx950;gfx942" pip3 install --no-build-isolation . -v

RUN pip uninstall transformer-engine transformer_engine -y 2>/dev/null || true

ARG TE_REPO=https://github.com/ROCm/TransformerEngine.git
ARG TE_COMMIT=5235bae2cc683a0ad4bf15221c746ab3c1e229e7
ARG AITER_COMMIT=c4b33df03faae1c4e470420d950f8a9589e9634d
RUN git clone --recursive "${TE_REPO}" TransformerEngine \
 && cd TransformerEngine \
 && git checkout "${TE_COMMIT}" \
 && git submodule update --init --recursive \
 && AITER_DIR=3rdparty/QoLA/3rdparty/aiter \
 && git -C "${AITER_DIR}" fetch origin "${AITER_COMMIT}" \
 && git -C "${AITER_DIR}" checkout "${AITER_COMMIT}" \
 && git -C "${AITER_DIR}" submodule update --init --recursive \
 && sed -i "s|^aiter_commit = .*|aiter_commit = \"${AITER_COMMIT}\"|" \
        transformer_engine/common/ck_fused_attn/qola_manifest.toml

ARG BWD_ATTN_ASM_ENABLE=1
RUN DEPS_DIR=/workspace/deps \
    TE_DIR=/workspace/deps/TransformerEngine \
    BWD_ATTN_ASM_ENABLE=${BWD_ATTN_ASM_ENABLE} \
    bash /workspace/Primus/primus/backends/transformer_engine/fwd_attn_asm/install.sh

ENV FMHA_HD64_ASM_CO=/workspace/deps/fwd-attn-asm/build/fwd_d64_opt128.co
ENV MLPERF_ENABLE_FWD_ATTN_ASM=1
ENV FMHA_HD64_ASM_LOG=0

RUN cd TransformerEngine && \
    NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_CK_FUSED_ATTN_PATH="" \
    NVTE_BUILD_MAX_JOBS=128 NVTE_FRAMEWORK=pytorch NVTE_ROCM_ARCH="gfx950;gfx942" \
    NVTE_USE_HIPBLASLT=1 PYTORCH_ROCM_ARCH="gfx950;gfx942" CU_NUM=304 \
    NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 \
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

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY primus_mllog-0.1.21-py3-none-any.whl /tmp/primus_mllog-0.1.21-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.21-py3-none-any.whl && rm /tmp/primus_mllog-0.1.21-py3-none-any.whl
