# GPT-OSS-20B Pretraining Benchmark

GPT-OSS 20B (Mixture of Experts) using Primus framework.

## Overview

This benchmark trains a 20B parameter GPT model with Mixture of Experts (MoE) architecture using the Primus framework on AMD GPUs.

# 1. Setup Docker Image

## Build Docker Image

Run the following build command from this directory. The build process will take a while to complete.

```bash
# From gpt-oss-20b/primus directory
docker build -t rocm/amd-mlperf:gpt_oss_20b_training_6.0 .
```

# 2. Prepare Dataset

The current codebase uses the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for training and evaluation.

## Download Preprocessed Data and Model

The pre-tokenized dataset and the tokenizer are available for download. Navigate to your desired download directory and run the following commands:

```bash
# Desired download directory
cd /data/gpt_oss_20b

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

Set the directory for data and results. Ensure `$LOGDIR` has write access.

```bash
export DATADIR=/data/gpt_oss_20b/data
export MODELDIR=/data/gpt_oss_20b/model
export LOGDIR=/data/gpt_oss_20b/results
export CONT=rocm/amd-mlperf:gpt_oss_20b_training_6.0

# Create results directory
mkdir -p $LOGDIR
sudo chmod -R 777 $LOGDIR
```

## Set Configuration

Set appropriate configuration and system-specific hyperparameters:

| Config File | System | Nodes × GPUs | Global Batch |
|-------------|--------|--------------|--------------|
| `config_MI350X_1x8x1_tp1pp1ep1_gbs32.sh` | MI350X | 1 × 8        | 32           |
| `config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh` | MI355X | 1 × 8        | 32           |
| `config_MI355X_2x8x1_tp1pp1ep1_gbs32.sh` | MI355X | 2 × 8        | 32           |
| `config_MI355X_4x8x1_tp1pp1ep1_gbs64.sh` | MI355X | 4 × 8        | 64           |

For single-node runs source the matching script before launching:

```bash
source config_MI355X_1x8x1_tp1pp1ep1_gbs32.sh
```

## Launch Training (Single Node)

### Single Run

```bash
export NEXP=1
bash run_with_docker.sh
```

### Multiple Runs (for submission)

```bash
export NEXP=10
bash run_with_docker.sh
```

After completion, logs will be available under `$LOGDIR`.

## Launch training (multi-node via slurm)

Use the `job.sh` in this directory to launch the training run (update `job.sh` per your cluster/run configuration).
```
sbatch job.sh
```

Note: For MI355X multi-node runs, we are using the AINIC bundle version 1.117.5-a-77.

# 4. Quality Metrics

## Quality Metric

Validation loss (log perplexity)

## Quality Target

Validation log perplexity = **3.34**

## Evaluation Frequency

Evaluation every **12,288 samples** (iteration count depends on GBS, e.g. 384 iters at GBS=32, 192 iters at GBS=64). The per-iter value is computed in each `config_*.sh` as `PRIMUS_EVAL_INTERVAL = EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE`.

## Evaluation Thoroughness

We evaluate using **1024 sequences** from the validation dataset.