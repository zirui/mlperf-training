ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Install Megatron-Bridge from bundled Primus
RUN cd /workspace/Primus/third_party/Megatron-Bridge && \
    pip install -e . --no-deps --no-build-isolation

# Apply Megatron-Bridge patches for LoRA SFT
COPY patches/ /tmp/patches/
RUN cd /workspace/Primus/third_party/Megatron-Bridge && \
    git config --global --add safe.directory /workspace/Primus && \
    git config --global --add safe.directory /workspace/Primus/third_party/Megatron-Bridge && \
    git reset --hard HEAD && \
    git apply /tmp/patches/megatron_nemo_lora_only.patch && \
    git apply /tmp/patches/megatron_bridge_validation_consumed_samples.patch && \
    git apply /tmp/patches/megatron_bridge_deterministic_eval.patch && \
    rm -rf /tmp/patches

# Install model preset and custom recipe not present in v26.2 base image
COPY src/llama2_70b_lora.yaml /workspace/Primus/primus/configs/models/megatron_bridge/llama2_70b_lora.yaml
RUN mkdir -p /workspace/Primus/primus/recipes \
    && mkdir -p /workspace/Primus/third_party/Megatron-Bridge/src/megatron/bridge/recipes/primus/recipes
COPY src/recipes__init__.py /workspace/Primus/primus/recipes/__init__.py
COPY src/llama2_custom.py /workspace/Primus/primus/recipes/llama2_custom.py
# Recipe must also be importable via megatron.bridge.recipes.primus.recipes.llama2_custom
COPY src/recipes__init__.py /workspace/Primus/third_party/Megatron-Bridge/src/megatron/bridge/recipes/primus/__init__.py
COPY src/recipes__init__.py /workspace/Primus/third_party/Megatron-Bridge/src/megatron/bridge/recipes/primus/recipes/__init__.py
COPY src/llama2_custom.py /workspace/Primus/third_party/Megatron-Bridge/src/megatron/bridge/recipes/primus/recipes/llama2_custom.py

WORKDIR /workspace/code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# MLPerf logging via primus_mllog (install before bulk COPY for layer caching)
COPY primus_mllog-0.1.3-py3-none-any.whl /tmp/primus_mllog-0.1.3-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.3-py3-none-any.whl && rm /tmp/primus_mllog-0.1.3-py3-none-any.whl

COPY . .
