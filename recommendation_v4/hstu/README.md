# Running AMD DLRMv4 Training PyTorch MLPerf Benchmark

Recommendation - DLRMv4 (HSTU ranking model on the Yambda-5b dataset).

# 1. Setup Docker Image

Build the submission image from `recommendation_v4/hstu`:

```bash
docker build --network=host -t rocm/amd-mlperf:dlrmv4_training_6.1 -f Dockerfile .
```

The commands below use that local tag. It is not a published image, so build it before running.

The Dockerfile specifies the pinned dependencies. FBGEMM-GPU has no public gfx950 build, so it is compiled from source; the build takes 30–60 minutes. `--build-arg MAX_JOBS=<n>` sets the compile parallelism.

# 2. Prepare Dataset

The current codebase uses the [Yambda-5b](https://huggingface.co/datasets/yandex/yambda) music-recommendation dataset from Yandex, preprocessed into the HSTU training cache.

## Download Preprocessed Data

The prepared dataset is available for download. You can navigate in the terminal to your desired download directory and run the following commands. In this example, we're using `/data/mlperf_dlrmv4` as the host download directory.

```bash
# desired download directory
mkdir -p /data/mlperf_dlrmv4
cd /data/mlperf_dlrmv4

# data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d data https://training.mlcommons-storage.org/metadata/dlrmv4_dataset.uri
```

After the download is complete, you should see the following under the data directory:
- Training cache: `processed_5b/hstu_cache_L4086/`
- Item popularity: `processed_5b/item_popularity.npy`
- Item mappings: `shared_metadata/album_item_mapping.parquet` and `shared_metadata/artist_item_mapping.parquet`

The data directory is ~227 GB. To verify every file against the published checksums, run from `recommendation_v4/hstu`:

```bash
DLRM_DATA_PATH=/data/mlperf_dlrmv4/data bash scripts/verify_dataset.sh
```

## Prebuild the Time-Window Index [Optional]

The first training run builds an ~18 GB time-window index in the data directory. To build it once ahead of time:

```bash
docker run --rm -v /data/mlperf_dlrmv4/data:/data rocm/amd-mlperf:dlrmv4_training_6.1 \
    python3 -m generative_recommenders.dlrm_v4.datasets.offline_ts_indices \
    --cache-dir /data/processed_5b/hstu_cache_L4086 \
    --history-length 4086 --min-history 4086 --window-seconds 86400
```

# 3. Run Training

### Set Environment

Set the directory for the data and results. Ensure that `$LOGDIR` has write access for the results to be written by running `sudo chmod -R 777 $LOGDIR`. In this example we use `/data/mlperf_dlrmv4/results` as the results directory, so please make sure to create this directory.

```bash
export DATADIR=/data/mlperf_dlrmv4/data
export LOGDIR=/data/mlperf_dlrmv4/results
export CONT=rocm/amd-mlperf:dlrmv4_training_6.1
```

### 1 node without Slurm

```bash
source config_MI355X_1x8x1.sh
export NEXP=1
bash run_with_docker.sh
```

For 10 trials, set `NEXP=10` instead of `NEXP=1`. MI350X uses `source config_MI350X_1x8x1.sh`.

### Slurm: 1, 2, or 4 nodes

`job.sh` allocates the nodes and runs `run_with_docker_slurm.sh` once on each of them. The file is set for 1 node. Edit it before submitting: set `--partition`, `CONT`, `DATADIR`, and `LOGDIR`. `CONT` must be an image every node can pull. `DATADIR` and `LOGDIR` are the directories from above.

`--nodes` and `DGXSYSTEM` are a pair:

- 1 node, global batch 8192: `--nodes=1` and `DGXSYSTEM=MI355X_1x8x1` (MI350X: `MI350X_1x8x1`)
- 2 nodes, global batch 16384: `--nodes=2` and `DGXSYSTEM=MI355X_2x8x1` (MI350X: `MI350X_2x8x1`)
- 4 nodes, global batch 32768: `--nodes=4` and `DGXSYSTEM=MI355X_4x8x1` (MI350X: `MI350X_4x8x1`)

`NEXP=10` is the submission. Set `NEXP=1` for one trial.

```bash
sbatch job.sh
```

Logs are written to `LOGDIR`.

Note: To optimize the machine's performance, the training script will also execute `runtime_tunables.sh` script before any training run.


# 4. Check Quality
### Quality metric

AUC of the `listen_plus` prediction on the held-out future time window.

### Quality target

Eval AUC = 0.75

### Evaluation frequency

We perform evaluation every **2,293,760** training samples (0.1% of the training stream): 280 steps at global batch 8192, 140 at 16384 and 70 at 32768. Evaluation starts after 2.5%, 3.03% and 4.1% of the stream at those three batch sizes.

### Evaluation thoroughness

Every evaluation scores the entire holdout window, **2,058,204** samples.
