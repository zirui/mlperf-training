# Flux1 Primus Training (NeMo-Free)

MLPerf Training benchmark for Flux1 (text-to-image diffusion transformer) using Primus + Megatron-core, without NeMo dependency.

## Architecture

This implementation uses Megatron-core's native `pretrain()` loop with the Flux model architecture extracted from NeMo v2.6.0. The NeMo wrapper layer (PyTorch Lightning, Hydra, NeMo data modules) is entirely eliminated.

**Stack:**
```
train.py → megatron.training.pretrain()
             ├── Flux model (Megatron-core VisionModule + TE layers)
             ├── Rectified-flow forward step (flux_forward_step.py)
             ├── megatron.energon datasets (flux_data.py)
             └── MLPerf logging (mlperf_callbacks.py)
```

## Quick Start

### 1. Build Docker image

```bash
# Submission image
docker build -t mlperf_flux1:v1 -f Dockerfile .

# Dev image (with RPD/TraceLens profiling)
bash dev/build_docker.sh mlperf_flux1:v1
```

### 2. Configure

Edit `config_MI355X_1x8x16.sh` for your system, or create a new config file:

```bash
export DGXSYSTEM=MI355X_1x8x16
export CONT=mlperf_flux1:v1
export DATADIR=/data/mlperf_flux1
export LOGDIR=/results/flux1
```

### 3. Run training

```bash
# Single-node, 8 GPUs
DGXSYSTEM=MI355X_1x8x16 \
CONT=mlperf_flux1:v1 \
DATADIR=/data/mlperf_flux1 \
LOGDIR=/results/flux1 \
bash run_with_docker.sh
```

### 4. Dev / Interactive

```bash
# Launch interactive dev container
DATADIR=/data/mlperf_flux1 bash dev/run_docker.sh

# Inside the container:
source config_MI355X_1x8x16.sh
bash run_and_time.sh
```

## Directory Structure

```
flux1/primus/
├── PLAN.md                      # Implementation plan
├── README.md                    # This file
├── Dockerfile                   # Submission image (Primus base, NO NeMo)
├── requirements.txt             # Minimal Python deps
├── run_with_docker.sh           # Docker-based launcher
├── run_and_time.sh              # Training entry with timing
├── runtime_tunables.sh          # Host tuning (THP, governor, etc.)
├── config_common.sh             # Shared defaults
├── config_MI355X_1x8x16.sh     # MI355X single-node config
├── conf/
│   └── flux1-pretrain-fp8.yaml  # Primus-format YAML config
├── src/
│   ├── train.py                 # Entry: Megatron pretrain()
│   ├── flux_model.py            # Flux + FluxConfig (Megatron-core VisionModule)
│   ├── flux_layers.py           # EmbedND, MLPEmbedder, TimeStepEmbedder (pure PyTorch)
│   ├── dit_layer_spec.py        # MMDiTLayer, FluxSingleTransformerBlock (Megatron-core + TE)
│   ├── dit_attention.py         # JointSelfAttention, FluxSingleAttention
│   ├── flux_forward_step.py     # Rectified-flow forward step + loss
│   ├── flux_validation.py       # 8-timestep validation logic
│   ├── flux_data.py             # megatron.energon dataset provider
│   ├── custom_weight_inits.py   # Weight initialization functions
│   └── mlperf_callbacks.py      # MLPerf logging
├── patches/                     # TE/Megatron patches
├── scripts/                     # Data preparation scripts
└── dev/
    ├── Dockerfile               # Dev image (RPD + TraceLens)
    ├── build_docker.sh          # Build helper
    ├── run_docker.sh            # Interactive container
    ├── run_and_time.sh          # Dev run with profiling
    └── run_with_docker_dev.sh   # Full dev pipeline
```

## Model Details

- **Flux Schnell**: 19 double-stream (MMDiT) layers + 38 single-stream layers
- **Hidden size**: 3072, **Attention heads**: 24
- **Training**: Rectified flow (linear interpolation noise schedule)
- **Precision**: FP8 hybrid via Transformer Engine
- **Target**: Validation loss ≤ 0.586

## Data

Preprocessed CC12M dataset with cached VAE latents and T5/CLIP text embeddings, stored in WebDataset format and loaded via `megatron.energon`.

See `scripts/` for data preparation utilities (shared with NeMo implementation).

## Differences from NeMo Implementation

| Feature | NeMo (`flux1/nemo/`) | Primus (`flux1/primus/`) |
|---|---|---|
| Framework | NeMo v2.6.0 + PyTorch Lightning | Megatron-core `pretrain()` directly |
| Config | Hydra YAML | Primus YAML + env vars |
| Docker image | ~15GB (NeMo + deps) | ~10GB (Primus base only) |
| Build time | ~25 min | ~10 min |
| Model code | NeMo wrappers around Megatron-core | Extracted standalone modules |
| Data loading | `DiffusionDataModule` (NeMo) | `megatron.energon` directly |
