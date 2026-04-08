ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Update Primus-Turbo to target commit on main branch
RUN cd /workspace/Primus-Turbo && \
    rm -rf build/ primus_turbo.egg-info/ 3rdparty/composable_kernel/ -rf && \
    git checkout . && \
    git checkout main && \
    git pull origin main && \
    git checkout c79e0f359a0c0952c634f3b48312cd6341b5d843 && \
    git submodule update --init --recursive

# Copy patch files and apply
WORKDIR /workspace/code/patches
COPY patches/megatron_validation_consumed_samples.patch .
COPY patches/te_fmha_backend_selection.patch .
COPY patches/te_native_swiglu.patch .
COPY patches/aiter_fused_rope.patch .

# Apply Megatron patches and reinstall
RUN cd /workspace/Primus/third_party/Megatron-LM && \
    git apply /workspace/code/patches/megatron_validation_consumed_samples.patch && \
    pip install -e . --no-deps --no-build-isolation

# Apply TE patches
RUN cd /opt/venv/lib/python3.12/site-packages && \
    patch -p1 < /workspace/code/patches/te_fmha_backend_selection.patch

RUN cd /opt/venv/lib/python3.12/site-packages && \
    patch -p1 < /workspace/code/patches/te_native_swiglu.patch || true

RUN cd /opt/venv/lib/python3.12/site-packages && \
    patch -p1 < /workspace/code/patches/aiter_fused_rope.patch || true

# Rebuild Primus-Turbo: uninstall aiter, clean build artifacts, reinstall
ARG GPU_ARCH=gfx950
RUN pip uninstall amd-aiter aiter -y && \
    cd /workspace/Primus-Turbo && \
    pip3 install -r requirements.txt && \
    PYTORCH_ROCM_ARCH="${GPU_ARCH}" GPU_ARCHS="${GPU_ARCH}" pip3 install --no-build-isolation . -v

WORKDIR /workspace/code

# megatron-energon for WebDataset loading
RUN pip install megatron-energon==7.1.0

# mlperf-common for MLPerf logging callbacks (mlperf-logging already in base image)
RUN pip install git+https://github.com/NVIDIA/mlperf-common.git@b86d175a05849d650a8ff69c1e2c37b9f4e61d51

# Minimal Python deps (NO NeMo)
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the current state of the code inside the image
COPY . .

# MLPerf logging via primus_mllog (replaces mlperf-logging + mlperf-common)
COPY primus_mllog-0.1.1-py3-none-any.whl /tmp/primus_mllog-0.1.1-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.1-py3-none-any.whl && rm /tmp/primus_mllog-0.1.1-py3-none-any.whl
