# MLPerf Training v6.0 — AMD Submission

AMD submissions for the [MLPerf Training v6.0](https://mlcommons.org/benchmarks/training/) benchmark suite, running on AMD Instinct GPUs with ROCm.

## Benchmarks

| Benchmark | Model | Target Hardware | Quality Target |
|---|---|---|---|
| [**Text-to-Image**](flux1/nemo/) | FLUX.1 Schnell | MI300X, MI355X (1–4 nodes) | Validation loss ≤ 0.586 |
| [**Small LLM MoE Pretraining**](small_llm_moe_pretraining/primus/) | GPT-OSS 20B MoE | MI355X | Validation loss ≤ 3.34 |

Each benchmark directory contains its own `README.md` with detailed setup, dataset preparation, and launch instructions.

## Software Stack

| Component | Version |
|---|---|
| ROCm | 7.2 |
| PyTorch | 2.9.1 |
| NeMo | v2.6.0 |
| Megatron-Core | ROCm fork |
| Transformer Engine | 2.8 (ROCm fork) |
| AITER | v0.1.12 |

## Quick Start

Each benchmark follows the same general workflow:

1. **Build** the Docker container (`docker build`)
2. **Download & preprocess** the dataset (instructions in each benchmark README)
3. **Source** the hardware config (`source config_MI355X_*.sh`)
4. **Launch** training (`bash run_with_docker.sh` or `sbatch job.sh` for multi-node)

Refer to the individual benchmark READMEs for full details:
- [FLUX.1 Text-to-Image](flux1/nemo/README.md)
- [GPT-OSS 20B MoE Pretraining](small_llm_moe_pretraining/primus/README.md)
