# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=rocm/primus:v26.6@sha256:4fcb3f210dc6675de0cb414cdc152184d1f43940d9a5f235e47d42cf644be122
FROM ${BASE_IMAGE}

ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
ARG PRIMUS_REF=bfc7db8e7a311c3b8d49ff6490824ebed2197953
ARG PRIMUS_TURBO_REPO=https://github.com/AMD-AGI/Primus-Turbo.git
ARG PRIMUS_TURBO_REF=728e15ff2b05c575c3193ee04faad84290e6adfe
ARG MLPERF_COMMON_REF=7771aeabc8b91402e95cb7a33f447d9d9941361f
ARG MAX_JOBS=96

WORKDIR /workspace

RUN rm -rf /workspace/Primus && \
    git clone --recursive "${PRIMUS_REPO}" /workspace/Primus && \
    cd /workspace/Primus && \
    git fetch origin "${PRIMUS_REF}" && \
    git checkout --detach "${PRIMUS_REF}" && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache

RUN python3 -m pip uninstall -y primus-mllog 2>/dev/null || true && \
    python3 -m pip install --no-cache-dir \
      "git+https://github.com/AMD-AGI/mlperf-common.git@${MLPERF_COMMON_REF}"

WORKDIR /workspace/deps

RUN cd /workspace/Primus/third_party/Megatron-LM && \
    pip install -e . --no-deps --no-build-isolation

RUN git clone --recursive "${PRIMUS_TURBO_REPO}" /workspace/deps/Primus-Turbo && \
    cd /workspace/deps/Primus-Turbo && \
    git fetch origin "${PRIMUS_TURBO_REF}" && \
    git checkout --detach "${PRIMUS_TURBO_REF}" && \
    git submodule update --init --recursive && \
    python3 -m pip install -r requirements.txt && \
    PRIMUS_TURBO_FRAMEWORK=PYTORCH \
    GPU_ARCHS=gfx950 \
    MAX_JOBS="${MAX_JOBS}" \
      python3 -m pip install --no-build-isolation --no-deps -v .

WORKDIR /workspace/code

COPY . .

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
