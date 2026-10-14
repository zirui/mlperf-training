# syntax=docker/dockerfile:1.7

# Pin the PyTorch/ROCm runtime by digest.
FROM rocm/primus:v26.7@sha256:68b7eb7d4db99ecfa18bd7972e5b6d8b78e56219f0a144a8ef5e9a993e57f195

# Delta from the baseline image: replace Primus and Primus-Turbo with the pinned FLUX
# revisions below; install the A6W4-to-FP8 switch; patch PyTorch, TorchAO, and AITER;
# replace primus-mllog; build the pinned RCCL fixes; and install the pinned ROCr runtime.
# The remaining Python/ROCm packages come unchanged from the baseline image.

# Build Primus with the integrated FLUX implementations.
ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
ARG PRIMUS_REF=94d8f575e8244214682cc64c317e8a4937a413c2
RUN rm -rf /workspace/Primus && \
    git clone --filter=blob:none "${PRIMUS_REPO}" /workspace/Primus && \
    cd /workspace/Primus && \
    git fetch origin "${PRIMUS_REF}" && \
    git checkout --detach "${PRIMUS_REF}" && \
    git submodule update --init --recursive --jobs 8
COPY patches/flux_fprop_switch.py /tmp/flux-build/flux_fprop_switch.py
RUN python3 /tmp/flux-build/flux_fprop_switch.py /workspace/Primus && \
    rm -f /tmp/flux-build/flux_fprop_switch.py

# Preinstall the pinned Primus diffusion requirements for offline launch.
RUN python3 -m pip install --no-cache-dir \
    -r /workspace/Primus/runner/helpers/hooks/train/pretrain/diffusion/requirements-diffusion.txt

# Build Primus-Turbo's Python and native layers from the same pinned commit.
ARG PRIMUS_TURBO_REPO=https://github.com/zirui/Primus-Turbo.git
ARG PRIMUS_TURBO_REVISION=e031c5bb3746e8b667676bd20e2697d75fe9f2b2
ARG MAX_JOBS=96
RUN rm -rf /workspace/Primus-Turbo && \
    git clone --filter=blob:none "${PRIMUS_TURBO_REPO}" /workspace/Primus-Turbo && \
    cd /workspace/Primus-Turbo && \
    git fetch origin "${PRIMUS_TURBO_REVISION}" && \
    git checkout --detach "${PRIMUS_TURBO_REVISION}" && \
    git submodule update --init --recursive --jobs 8 && \
    (python3 -m pip uninstall -y primus-turbo || true) && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    PRIMUS_TURBO_FRAMEWORK=PYTORCH \
    GPU_ARCHS=gfx950 \
    MAX_JOBS="${MAX_JOBS}" \
      python3 -m pip install --no-build-isolation --no-deps -v . && \
    python3 tools/check_flux_install.py "${PRIMUS_TURBO_REVISION}"

LABEL org.opencontainers.image.source="https://github.com/AMD-AGI/Primus-Turbo" \
      org.opencontainers.image.revision="${PRIMUS_TURBO_REVISION}"

# Patch PyTorch's selective FP8 GEMM dispatch and unsupported Inductor fusion fallback.
COPY scripts/pytorch/ /tmp/flux-build/pytorch/
RUN python3 /tmp/flux-build/pytorch/patch_selective_fp8_gemm.py && \
    python3 /tmp/flux-build/pytorch/patch_inductor_fusion.py && \
    python3 /tmp/flux-build/pytorch/check_install.py && \
    rm -rf /tmp/flux-build/pytorch

# Patch TorchAO's forward-cast reuse and natural-layout weight-gradient dispatch.
COPY scripts/torchao/ /tmp/flux-build/torchao/
RUN python3 /tmp/flux-build/torchao/patch_float8_linear.py && \
    python3 /tmp/flux-build/torchao/check_install.py && \
    rm -rf /tmp/flux-build/torchao

# Keep AITER's FP32 atomic attention path selectable at runtime.
COPY scripts/aiter/ /tmp/flux-build/aiter/
RUN python3 /tmp/flux-build/aiter/patch_attention.py && \
    python3 /tmp/flux-build/aiter/check_install.py && \
    rm -rf /tmp/flux-build/aiter

# Replace the base image's primus-mllog with pinned mlperf-common.
ARG MLPERF_COMMON_REF=7771aeabc8b91402e95cb7a33f447d9d9941361f
RUN (python3 -m pip uninstall -y primus-mllog 2>/dev/null || true) && \
    python3 -m pip install --no-cache-dir \
      "git+https://github.com/AMD-AGI/mlperf-common.git@${MLPERF_COMMON_REF}"

# Build the integrated RCCL fixes against this image's ROCm libraries.
ARG RCCL_REPO=https://github.com/ROCm/rocm-systems.git
ARG RCCL_REV=d75507948fa94106ea2adf5c158af31e6fbda6ef
ARG RCCL_MAX_JOBS=96
RUN set -eux; \
    SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel; \
    export PATH="${SDK}/bin:/opt/venv/bin:${PATH}" ROCM_PATH="${SDK}"; \
    git clone --filter=blob:none --sparse "${RCCL_REPO}" /tmp/rccl-src; \
    git -C /tmp/rccl-src sparse-checkout set projects/rccl; \
    git -C /tmp/rccl-src fetch origin "${RCCL_REV}"; \
    git -C /tmp/rccl-src checkout --detach "${RCCL_REV}"; \
    grep -q "symAvailable && cfgAllowsSymk" /tmp/rccl-src/projects/rccl/src/scheduler/symmetric_sched.cc; \
    grep -q "static const int runtimeSupported" /tmp/rccl-src/projects/rccl/src/misc/rocmwrap.cc; \
    mkdir -p /tmp/rccl-build && cd /tmp/rccl-build; \
    cmake -G Ninja /tmp/rccl-src/projects/rccl \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER="${SDK}/bin/amdclang++" \
        -DCMAKE_C_COMPILER="${SDK}/bin/amdclang" \
        -DCMAKE_PREFIX_PATH="${SDK}" \
        -DROCM_PATH="${SDK}" \
        -DGPU_TARGETS=gfx950 \
        -DBUILD_TESTS=OFF \
        -DROCTX=ON; \
    ninja -j "${RCCL_MAX_JOBS}"; \
    DST=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/librccl.so.1; \
    test -f "${DST}"; \
    cp -L /tmp/rccl-build/librccl.so.1 "${DST}"; \
    test "$(strings "${DST}" | grep -c 'Hierarchical CE')" -gt 0; \
    readelf -d "${DST}" | grep -E 'NEEDED.*(amd_smi|amdhip64|hsa-runtime)'; \
    rm -rf /tmp/rccl-src /tmp/rccl-build

LABEL com.amd.mlperf.rccl.source="${RCCL_REPO}" \
      com.amd.mlperf.rccl.revision="${RCCL_REV}"

# Install the checksum-pinned ROCr nightly that avoids the RCCL SDMA VMemoryPtrInfo crash.
ARG ROCR_SHA256=0ab5653be3b2bf2802c6a36cd4092d6191a985adf380d40dc55f9cfceb2c27d9
COPY rocr/libhsa-runtime64.so.1 /tmp/rocr/libhsa-runtime64.so.1
RUN set -eux; \
    echo "${ROCR_SHA256}  /tmp/rocr/libhsa-runtime64.so.1" | sha256sum -c -; \
    DST=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1; \
    test -f "${DST}"; \
    cp /tmp/rocr/libhsa-runtime64.so.1 "${DST}"; \
    chmod 0755 "${DST}"; \
    rm -rf /tmp/rocr; \
    echo "${ROCR_SHA256}  ${DST}" | sha256sum -c -; \
    readelf -d "${DST}" | grep -E 'SONAME.*hsa-runtime'

LABEL com.amd.mlperf.rocr.source="rocr-nightly-0929 (primus_rocm10.2_20260929)" \
      com.amd.mlperf.rocr.sha256="${ROCR_SHA256}"

WORKDIR /workspace/code
COPY . .

CMD ["/usr/bin/bash"]
