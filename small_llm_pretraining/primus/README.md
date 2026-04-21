# LLama3.1 8B Pretraining Benchmark

LLama3.1 8B pretraining using the Primus framework on AMD MI355X GPUs.

**Key Features:**
- 8B parameter LLama3.1 dense model
- FP8 hybrid precision training
- Megatron backend via Primus

# 1. Setup Docker Image

## Option 1: Build Docker Image

Run the following build command from this directory:

```bash
docker build -t rocm/amd-mlperf:llama3.1_8b_training_6.0 .
```

## Option 2: Build Dev Image (with profiling tools)

```bash
bash dev/build_docker.sh
```

This builds the base submission image and a dev image with RPD profiler and TraceLens.

# 2. Prepare Dataset

The current codebase uses the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for training and evaluation.

## Download Preprocessed Data and Model

```bash
cd /data/mlperf_llama31_8b

# Download training and validation data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# Download model tokenizer
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri

mv llama3_1_8b_tokenizer model
```

After download, you should see files with the following naming conventions:
- Training: `c4-train.en_6_text_document.bin` and `.idx`
- Validation: `c4-validation-91205-samples.en_text_document.bin` and `.idx`

The data directory is approximately **80 GB** and model directory is approximately **30 GB**.

# 3. Run Training

## Set Environment Variables

```bash
export DATADIR=/data/mlperf_llama31_8b/data
export MODELDIR=/data/mlperf_llama31_8b/model
export LOGDIR=/data/mlperf_llama31_8b/results
export CONT=rocm/amd-mlperf:llama3.1_8b_training_6.0

mkdir -p $LOGDIR
sudo chmod -R 777 $LOGDIR
```

## Set Configuration

```bash
source config_MI355X_1x8x1.sh
```

## Launch Training

### Single Run

```bash
export NEXP=1
export CONFIG_NAME=config_MI355X_1x8x1.sh
bash run_with_docker.sh
```

### Multiple Runs (for submission)

```bash
export NEXP=10
bash run_with_docker.sh
```

After completion, logs will be available under `$LOGDIR`.

# 4. Quality Metrics

## Quality Metric

Validation loss (log perplexity)

## Quality Target

Validation log perplexity = **3.3**

## Evaluation Frequency

Evaluation every **384 iterations** (12,288 samples with GBS=32)

## Evaluation Thoroughness

We evaluate using **1024 sequences** from the validation dataset.

# 5. Model Architecture

| Parameter | Value |
|-----------|-------|
| Model Size | 8B parameters |
| Architecture | LLama3.1 (dense) |
| Sequence Length | 8192 |
| Precision | BF16 / FP8 hybrid |
| Hidden Size | 4096 |
| Layers | 32 |
| Attention Heads | 32 |

# 6. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Micro Batch Size | 2 |
| Global Batch Size | 32 |
| Learning Rate | 8e-4 |
| LR Schedule | Cosine decay with warmup |
| Weight Decay | 0.1 |
| Adam b1, b2 | 0.9, 0.95 |
| Training Iterations | 1,200,000 |

# 7. Directory Structure

```
small_llm_pretraining/primus/
├── conf/                               # Configuration files
│   └── llama3.1_8B-pretrain-fp8.yaml
├── dev/                                # Development scripts
│   ├── Dockerfile
│   ├── build_docker.sh
│   ├── run_docker.sh
│   ├── run_and_time.sh
│   └── run_with_docker_dev.sh
├── src/                                # Training source code
│   └── train.py
├── config_MI355X_1x8x1.sh             # System configuration
├── Dockerfile
├── run_and_time.sh
├── run_with_docker.sh
├── runtime_tunables.sh
├── requirements.txt
└── primus_mllog-0.1.1-py3-none-any.whl
```
