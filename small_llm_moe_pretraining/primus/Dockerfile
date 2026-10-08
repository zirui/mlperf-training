# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=rocm/primus:v26.6@sha256:4fcb3f210dc6675de0cb414cdc152184d1f43940d9a5f235e47d42cf644be122
FROM ${BASE_IMAGE}

ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
ARG PRIMUS_REF=78287080eb3dbf7289c76b2adceb0461811f5e1a
ARG PRIMUS_TURBO_REF=30e6de8781759063597a62bcbc01a96dfa8e02c6
ARG MLPERF_COMMON_REF=7771aeabc8b91402e95cb7a33f447d9d9941361f
ARG RCCL_REPO=https://github.com/ROCm/rocm-systems.git
ARG RCCL_REF=dde68f300744982643bb371f7c3e165e8d61d728
ARG MAX_JOBS=96

WORKDIR /workspace

RUN rm -rf /workspace/Primus && \
    git init /workspace/Primus && \
    cd /workspace/Primus && \
    git remote add origin "${PRIMUS_REPO}" && \
    git fetch --depth 1 origin "${PRIMUS_REF}" && \
    git checkout --detach FETCH_HEAD && \
    git submodule update --init --recursive --depth 1 third_party/Megatron-LM

RUN set -eux; \
    source_root=/tmp/rocm-systems-rccl-ce; \
    git init "$source_root"; \
    git -C "$source_root" remote add origin "$RCCL_REPO"; \
    git -C "$source_root" sparse-checkout init --cone; \
    git -C "$source_root" sparse-checkout set projects/rccl; \
    timeout --kill-after=30s 10m git -C "$source_root" fetch --depth 1 origin "$RCCL_REF"; \
    git -C "$source_root" checkout --detach FETCH_HEAD; \
    test "$(git -C "$source_root" rev-parse HEAD)" = "$RCCL_REF"; \
    if [ ! -e /opt/rocm ]; then ln -sfn "$ROCM_PATH" /opt/rocm; fi; \
    cd "$source_root/projects/rccl"; \
    ./install.sh \
      --install \
      --amdgpu_targets gfx950 \
      --jobs "$MAX_JOBS" \
      --prefix /opt/rocm; \
    built_rccl="$(readlink -f /opt/rocm/lib/librccl.so)"; \
    built_sha="$(sha256sum "$built_rccl" | awk '{print $1}')"; \
    find /opt /usr \( -type f -o -type l \) -name 'librccl.so.1*' -print | \
      while IFS= read -r candidate; do \
        runtime_rccl="$(readlink -f "$candidate")"; \
        [ -n "$runtime_rccl" ] || continue; \
        if [ "$runtime_rccl" != "$built_rccl" ]; then \
          install -m 0755 "$built_rccl" "$runtime_rccl"; \
        fi; \
        test "$(sha256sum "$runtime_rccl" | awk '{print $1}')" = "$built_sha"; \
      done; \
    rm -rf "$source_root"

RUN python3 -m pip uninstall -y primus-mllog 2>/dev/null || true && \
    python3 -m pip install --no-cache-dir \
      "git+https://github.com/AMD-AGI/mlperf-common.git@${MLPERF_COMMON_REF}"

WORKDIR /workspace/deps

RUN git clone --recursive https://github.com/AMD-AGI/Primus-Turbo.git && \
    cd Primus-Turbo && \
    git checkout --detach "${PRIMUS_TURBO_REF}" && \
    git submodule update --init --recursive && \
    python3 -m pip install -r requirements.txt && \
    PRIMUS_TURBO_FRAMEWORK=PYTORCH \
    GPU_ARCHS=gfx950 \
    MAX_JOBS="${MAX_JOBS}" \
      python3 -m pip install --no-build-isolation --no-deps -v . && \
    cd .. && rm -rf Primus-Turbo /workspace/Primus-Turbo

WORKDIR /workspace/code

COPY . .

RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    git -C /workspace/Primus/third_party/Megatron-LM apply \
      /workspace/code/patches/router-bwd-bf16.patch
