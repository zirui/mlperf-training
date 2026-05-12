ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Install library dependencies
WORKDIR /workspace

## Unset NVTE_CK_FUSED_ATTN_PATH from base image so TE's cmake uses prebuilt download
ENV NVTE_CK_FUSED_ATTN_PATH=""

# TE
RUN pip uninstall -y transformer-engine
ARG TE_COMMIT=261eda6ab09e603d816af78d31ba793f643f28fd
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    && cd TransformerEngine && git checkout ${TE_COMMIT} \
    && git submodule sync \
    && git submodule update --init --recursive

RUN cd TransformerEngine \
    && NVTE_PROJECT_BUILDING=1 NVTE_USE_ROCM=1 NVTE_FUSED_ATTN_AOTRITON=0 \
    NVTE_ROCM_ARCH="gfx950;gfx942" GPU_ARCHS="gfx950;gfx942" \
    NVTE_FRAMEWORK='pytorch' \
    NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH="gfx950;gfx942" CU_NUM=304 \
    pip install --no-build-isolation -e . -v \
    && rm -rf /root/.cache

# AITER
ARG AITER_COMMIT=77455e3ecf4f0d28756afc452e914940c45b944b
RUN rm -rf /workspace/aiter \
    && git clone --recursive https://github.com/ROCm/aiter.git \
    && cd aiter && git checkout ${AITER_COMMIT}
ENV PYTHONPATH="/workspace/aiter:${PYTHONPATH}"
RUN echo "/workspace/aiter" > \
    $(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/aiter_workspace.pth

# Pre-compile AITER JIT modules
RUN pip install psutil
COPY precompile_aiter.py /tmp/precompile_aiter.py
RUN python3 /tmp/precompile_aiter.py

# Primus
ARG PRIMUS_REPO=https://github.com/AMD-AGI/Primus.git
ARG PRIMUS_BRANCH=689ae9dcc77c7cb7b2ca05a0a7d8993199c38ba4

RUN rm -rf /workspace/Primus && \
    git clone --recurse-submodules ${PRIMUS_REPO} /workspace/Primus && \
    cd /workspace/Primus && \
    git checkout ${PRIMUS_BRANCH} && \
    git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache

# Apply Megatron-Bridge patches
COPY patches/ /tmp/patches/
RUN cd /workspace/Primus/third_party/Megatron-Bridge && \
    git config --global --add safe.directory /workspace/Primus && \
    git config --global --add safe.directory /workspace/Primus/third_party/Megatron-Bridge && \
    git apply /tmp/patches/megatron_nemo_lora_only.patch && \
    git apply /tmp/patches/megatron_bridge_validation_consumed_samples.patch && \
    git apply /tmp/patches/megatron_bridge_deterministic_eval.patch && \
    git apply /tmp/patches/sft_attention_mask_cache.patch

RUN cd /workspace/Primus/third_party/Megatron-Bridge/3rdparty/Megatron-LM && \
    git apply /tmp/patches/megatron_lm_mxfp4_recipe.patch && \
    rm -rf /tmp/patches

# Install Megatron-Bridge editable so patched source is used at runtime
RUN cd /workspace/Primus/third_party/Megatron-Bridge && \
    pip install -e . --no-deps --no-build-isolation

# Pre-install Megatron-Bridge hook dependencies so the runtime hook is a no-op
RUN pip install "onnx==1.20.0rc1" "datasets>=2.14.0" "transformers==4.57.6" \
    "qwen-vl-utils" "timm" "open-clip-torch>=3.2.0" "flash-linear-attention" \
    "megatron-energon" "bitstring" "filetype" \
    && pip install nvidia_resiliency_ext || true \
    && rm -rf /root/.cache

# Stub modelopt package: Megatron-Bridge imports modelopt.torch.* at module level
# for unused features.
RUN mkdir -p /opt/venv/lib/python3.12/site-packages/modelopt/torch/distill/plugins \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/opt/plugins \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/quantization \
    && touch /opt/venv/lib/python3.12/site-packages/modelopt/__init__.py \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/__init__.py \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/distill/__init__.py \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/distill/plugins/__init__.py \
             /opt/venv/lib/python3.12/site-packages/modelopt/torch/quantization/__init__.py \
    && printf '%s\n' \
       'def get_tensor_shapes_adjust_fn_for_distillation(*a, **kw): return None' \
       > /opt/venv/lib/python3.12/site-packages/modelopt/torch/distill/plugins/megatron.py \
    && printf '%s\n' \
       'def save_modelopt_state(*a, **kw): pass' \
       'def save_sharded_modelopt_state(*a, **kw): pass' \
       'def restore_sharded_modelopt_state(*a, **kw): pass' \
       'def restore_modelopt_state(*a, **kw): pass' \
       > /opt/venv/lib/python3.12/site-packages/modelopt/torch/opt/__init__.py \
    && cp /opt/venv/lib/python3.12/site-packages/modelopt/torch/opt/__init__.py \
          /opt/venv/lib/python3.12/site-packages/modelopt/torch/opt/plugins/__init__.py

# Install model preset and custom recipe
COPY src/llama2_70b_lora.yaml /workspace/Primus/primus/configs/models/megatron_bridge/llama2_70b_lora.yaml
COPY src/llama2_custom.py /workspace/Primus/primus/recipes/llama2_custom.py
COPY src/_log_suppression.py /workspace/Primus/primus/recipes/_log_suppression.py

WORKDIR /workspace/code

COPY . .

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#Install primus_mllog for MLPerf logging
COPY primus_mllog-0.1.21-py3-none-any.whl /tmp/primus_mllog-0.1.21-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.21-py3-none-any.whl && rm /tmp/primus_mllog-0.1.21-py3-none-any.whl
