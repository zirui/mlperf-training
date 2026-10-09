# Flux1 Primus Training

MLPerf Training benchmark for Flux1 (text-to-image diffusion transformer) on the
Primus diffusion backend.

## Build and run

To build the container and run the training, follow these steps:

```bash
cd flux1/primus
DOCKER_BUILDKIT=1 docker build -t mlperf_flux1:integrated -f Dockerfile .
```


## Data

Download once to shared storage that every node can read:

```bash
PREPROCESSED_PATH=/shared/path/to/flux1/data bash scripts/download_ref_data.sh
```

`DATADIR` must contain the three downloaded directories, which training reads as-is:

```
$DATADIR/
├── cc12m_preprocessed
├── coco_preprocessed
└── empty_encodings
```


## Run (4 nodes)

Edit the `#SBATCH` header in `job.sh` (partition, and nodelist if needed), then:

```bash
export CONT=mlperf_flux1:integrated
export DATADIR=/path/to/flux1/data
export LOGDIR=/path/to/results
sbatch job.sh         # default config: 4 nodes
```

The default node config is 4 nodes - `config_MI355X_4x8x32.sh`. For single node, set `DGXSYSTEM=MI355X_1x8x32`.

## Directory Structure

```
flux1/primus/
├── Dockerfile
├── job.sh
├── config_MI355X_4x8x32.sh
├── config_MI355X_1x8x32.sh
├── config_common.sh
├── run_and_time.sh
├── run_with_docker_slurm.sh
├── run_with_docker.sh
├── scripts/
├── rocr/libhsa-runtime64.so.1
└── conf/flux.1_schnell_t2i-pretrain.yaml
```

## Model Details

- **Flux Schnell**: 19 double-stream (MMDiT) layers + 38 single-stream layers
- **Hidden size**: 3072, **Attention heads**: 24
- **Target**: Validation loss ≤ 0.586
