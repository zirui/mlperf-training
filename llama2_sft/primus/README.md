# LLama2 70B LoRA SFT Benchmark (Primus)

LLama2 70B LoRA supervised fine-tuning using the Primus framework with Megatron-Bridge on AMD MI355X GPUs.

**Key Features:**
- 70B parameter LLama2 model with LoRA adapters
- FP8 hybrid precision training via Megatron-Bridge
- Packed sequence support for efficient SFT
- Deterministic evaluation with capped validation samples

## 1. Setup Docker Image

### Build Docker Image

```bash
docker build -t rocm/amd-mlperf:llama2_70b_sft_primus_6.0 .
```

### Build Dev Image (with profiling tools)

```bash
bash dev/build_docker.sh
```

## 2. Prepare Dataset

The dataset uses packed `.npy` format for efficient LoRA SFT training.

Required files in the data directory (`${DATADIR}/data/`):
- `train.npy` — packed training data
- `validation.npy` — packed validation data
- `packed_metadata.jsonl` — sequence metadata

Required files in the model directory (`${DATADIR}/model/`):
- `megatron_checkpoints/Llama-2-70b-hf/iter_0000000` — pretrained Megatron-format checkpoint

This follows the same directory layout as the nemo implementation:
```
/data/mlperf_llama2/
├── data/          # train.npy, validation.npy, packed_metadata.jsonl
└── model/         # megatron_checkpoints/Llama-2-70b-hf/iter_0000000
```

## 3. Run Training

### Set Environment Variables

```bash
export DATADIR=/data/mlperf_llama2
export MODEL=/data/mlperf_llama2/model
export LOGDIR=/data/mlperf_llama2/results
export CONT=rocm/amd-mlperf:llama2_70b_sft_primus_6.0

mkdir -p $LOGDIR
```

### Launch Training

```bash
export NEXP=1
export CONFIG_NAME=config_MI355X_1x8x1.sh
bash run_with_docker.sh
```

## 4. Quality Metrics

- **Task:** LoRA supervised fine-tuning
- **Training iterations:** 1024
- **Evaluation interval:** Every 48 iterations
- **Evaluation samples:** 24 iterations x GBS

## 5. Model Architecture

| Parameter | Value |
|-----------|-------|
| Model Size | 70B parameters |
| PEFT Method | LoRA |
| Sequence Length | 8192 |
| Precision | BF16 / FP8 hybrid |
| Hidden Size | 8192 |
| Layers | 80 |
| Attention Heads | 64 |

## 6. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Micro Batch Size | 1 |
| Global Batch Size | 8 |
| Learning Rate | 4e-4 |
| LR Schedule | Cosine decay |
| Weight Decay | 0.0001 |
| Adam b1, b2 | 0.9, 0.999 |
| Gradient Clipping | 0.3 |

## 7. Directory Structure

```
llama2_sft/primus/
├── conf/                                # Configuration files
│   └── llama2_70b_lora_posttrain.yaml
├── patches/                             # Megatron-Bridge patches
│   ├── megatron_nemo_lora_only.patch
│   ├── megatron_bridge_validation_consumed_samples.patch
│   └── megatron_bridge_deterministic_eval.patch
├── dev/                                 # Development scripts
│   ├── Dockerfile
│   ├── build_docker.sh
│   ├── run_docker.sh
│   ├── run_and_time.sh
│   └── run_with_docker_dev.sh
├── config_MI355X_1x8x1.sh              # System configuration
├── Dockerfile
├── run_and_time.sh
├── run_with_docker.sh
├── runtime_tunables.sh
├── requirements.txt
└── README.md
```
