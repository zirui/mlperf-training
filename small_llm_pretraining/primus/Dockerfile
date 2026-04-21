ARG BASE_IMAGE=docker.io/rocm/primus:v26.2
FROM ${BASE_IMAGE}

# Install Megatron-LM from bundled Primus
RUN cd /workspace/Primus/third_party/Megatron-LM && \
    pip install -e . --no-deps --no-build-isolation

WORKDIR /workspace/code

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

COPY primus_mllog-0.1.1-py3-none-any.whl /tmp/primus_mllog-0.1.1-py3-none-any.whl
RUN pip install /tmp/primus_mllog-0.1.1-py3-none-any.whl && rm /tmp/primus_mllog-0.1.1-py3-none-any.whl
