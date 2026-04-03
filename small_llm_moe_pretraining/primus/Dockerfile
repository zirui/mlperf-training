ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Install library dependencies
WORKDIR /workspace/deps

ADD patches /workspace/deps/patches

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

# Transformer Engine (TE 2.8 + MXFP4)
RUN pip uninstall transformer-engine transformer_engine -y 2>/dev/null || true

ARG TE_COMMIT=6f406e55cd2ab7590d8e593c71350a77d2c930c2
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

# CUDAGuard fix for MXFP4 standalone quantization on multi-GPU
RUN cd TransformerEngine && git apply /workspace/deps/patches/mxfp4_hip_cudaguard.patch

# FMHA backend selection (gpt-oss-20b specific)
RUN cd TransformerEngine && git apply /workspace/deps/patches/te_fmha_backend_selection.patch

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
    NVTE_ROCM_ARCH="gfx950" GPU_ARCHS="gfx950" \
    NVTE_FRAMEWORK='pytorch' \
    NVTE_AITER_PREBUILT_BASE_URL="${PREBUILT_URL}" \
    NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx950" CU_NUM=304 \
    pip install --no-build-isolation -e . -v

# AITER - Required for MXFP4 quantization
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
    GPU_ARCHS="gfx950" python3 -c "\
import os; os.environ['GPU_ARCHS'] = 'gfx950'; \
from aiter.jit.core import get_args_of_build, build_module; \
[build_module(m, **{k: v for k, v in get_args_of_build(m).items() if k not in ('md_name', 'hip_clang_path')}) \
  for m in ('module_rope_general_fwd','module_rope_general_bwd','module_gemm_a4w4_asm')]; \
print('AITER RoPE+FP4 GEMM pre-compiled OK')"

# MXFP4 weight cache + lazy columnwise quant
# AITER fused RoPE: dispatch to aiter.ops.rope when available
RUN cd TransformerEngine && git apply /workspace/deps/patches/mxfp4_te_modules.patch \
    && git apply /workspace/deps/patches/aiter_fused_rope.patch

# Megatron-LM
RUN cd /workspace/Primus/third_party/Megatron-LM && \
    git apply /workspace/deps/patches/megatron_validation_consumed_samples.patch && \
    pip install -e . --no-deps --no-build-isolation

WORKDIR /workspace/code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the current state of the code inside the image
COPY . .

COPY primus_mllog-0.1.1-py3-none-any.whl /tmp/primus_mllog-0.1.1-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.1-py3-none-any.whl && rm /tmp/primus_mllog-0.1.1-py3-none-any.whl

# AINIC driver (Pensando ionic NIC support for multi-node RDMA, optional)
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
