ARG BASE_IMAGE=rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.10.0
FROM ${BASE_IMAGE}

WORKDIR /workspace

RUN pip install pybind11
RUN pip install ninja
RUN pip install packaging
RUN /usr/bin/python3 -m pip install pyYAML

# Install library dependencies
WORKDIR /workspace/deps

ADD patches /workspace/deps/patches

# Megatron-core mpo branch
ARG MEGATRON_COMMIT=f312f7db1ec26b30831aae0fac023c6895a4ac1e
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
RUN cd megatron_lm && git checkout ${MEGATRON_COMMIT}   \
    && pip install -e . && cd megatron/core/datasets && make

ENV PYTHONPATH "${PYTHONPATH}:/workspace/deps/megatron_lm"

# NeMo - keep ROCm torch, use constraints
RUN pip freeze | grep "torch" > /tmp/torch-constraints.txt || echo "torch" > /tmp/torch-constraints.txt && \
    cat /tmp/torch-constraints.txt

# NeMo v2.6.0 (2025-12-03)
RUN git clone https://github.com/NVIDIA/NeMo nemo \
    && cd nemo && git checkout v2.6.0 \
    && sed -i '12d' requirements/requirements_nlp.txt
RUN cd /workspace/deps/nemo \
    && git apply /workspace/deps/patches/diffusion_vae_blocks_logging.patch \
    && PIP_CONSTRAINT=/tmp/torch-constraints.txt pip install --no-build-isolation -e ".[nlp,multimodal,llm]"

# NeMo-Run
RUN pip install git+https://github.com/NVIDIA/NeMo-Run.git@v0.4.0

# Python deps
# Important: install after NeMo so transformers pin is overridden to the version we need
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python -m ensurepip || apt-get install -y python3-venv \
&& pip install cmake==3.28.3

# ROCm Transformer Engine
ARG TE_COMMIT=261eda6ab09e603d816af78d31ba793f643f28fd
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

RUN cd TransformerEngine \
    && NVTE_PROJECT_BUILDING=1 NVTE_USE_ROCM=1 NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_ROCM_ARCH="gfx942;gfx950" GPU_ARCHS="gfx942;gfx950" \
    NVTE_FRAMEWORK='pytorch' \
    NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx942;gfx950" CU_NUM=304 \
    pip install --no-build-isolation -e . -v

# Uninstall bitsandbytes to remove NVIDIA related warnings
RUN pip uninstall -y bitsandbytes mlperf-logging megatron-energon mlperf-common

RUN pip install megatron-energon==7.1.0
RUN pip install git+https://github.com/mlcommons/logging.git@6.0.0-rc5
RUN pip install git+https://github.com/NVIDIA/mlperf-common.git@b86d175a05849d650a8ff69c1e2c37b9f4e61d51

WORKDIR /workspace/code

# Copy the current state of the code inside the image
COPY . .
