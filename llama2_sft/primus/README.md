# Running AMD LLama2-70B LoRA PyTorch MLPerf Benchmark (Primus)

This benchmark represents a LLama2-70B LoRA finetuning on the [GovReport](https://gov-report-data.github.io/) dataset.

## Prerequisites

- Access to the `meta-llama/Llama-2-70b-hf` model on HuggingFace
(requires signing the LLAMA 2 COMMUNITY LICENSE AGREEMENT)
- A HuggingFace access token (`HF_TOKEN`)
- ~270 GB free disk space for model download + conversion

## Step 1: Build the Docker Image

Run the following build command from the root of the repository. The build process will take a while to complete. Ensure that all scripts have write access by running `sudo chmod -R 777'.

```bash
docker build -t rocm/amd-mlperf:llama2_70b_training_6.0 .
```

## Step 2: Prepare Dataset

## General Information

GovReport is a dataset for long document summarization that consists of reports written by government research agencies. The dataset hosted on the MLPerf drive is already tokenized and packed so that each sequence has length 8192.

The used model is the LLama2-70B with fused QKV. You will need 270GB to download and convert the model.

## Download and Preprocess Data & Model

To download the model from Huggingface, you'll need to sign the LLAMA 2 COMMUNITY LICENSE AGREEMENT as well as obtain a Hugginface Token `HF_TOKEN`.

Start the docker container by mounting the volume you want to use for downloading the data under `/data` within the container. In this example we use `/data/mlperf_llama2/data` as the host download directory:

```bash
docker run -it -v /data/mlperf_llama2/data:/data \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    rocm/amd-mlperf:llama2_70b_training_6.0
```

Start the script for downloading and preprocessing data from within the container (HF_TOKEN is needed to download the model weights):

```bash
export HF_TOKEN=<your-huggingace-token>
bash ./scripts/prepare_data_and_model.sh
```

## Verify Data

After completion of the above step, your data directory should look like this:

```
/data/mlperf_llama2/data/
├── train.npy                          # Packed training data (seq_length=8192)
├── validation.npy                     # Packed validation data
├── packed_metadata.jsonl              # Sequence metadata for Megatron-Bridge
├── megatron_checkpoints/
│   └── Llama-2-70b-hf/
│       ├── latest_checkpointed_iteration.txt
│       ├── latest_train_state.pt
│       └── iter_0000000/
│           ├── __0_0.distcp           # Sharded weights (~64 GB)
│           ├── __0_1.distcp           # Sharded weights (~64 GB)
│           ├── common.pt
│           ├── .metadata
│           ├── metadata.json
│           ├── run_config.yaml
│           ├── modelopt_run_config.yaml
│           ├── train_state.pt
│           └── tokenizer/
│               ├── tokenizer.model
│               ├── tokenizer_config.json
│               └── special_tokens_map.json

```

---

## Exit Container

Exit the container by running the below command

```bash
exit
```

# 3. Run Training

### Set Environment

Set the directory for the data, model and results. Ensure that `$LOGDIR` has write access for the results to be written by running `sudo chmod -R 777 $LOGDIR`, In this example we use `/data/mlperf_llama2/results` as the results directory, so please make sure to create this directory.

```bash
export DATADIR=/data/mlperf_llama2
export LOGDIR=/data/mlperf_llama2/results
export CONT=rocm/amd-mlperf:llama2_70b_training_6.0
```

### Set Configuration

Source the appropriate config for your GPU:

```bash
source config_MI355X_1x8x1.sh 
```

### Launch 1 Training Run

If you want to perform a single run, use:

```bash
export NEXP=1
bash run_with_docker.sh
```

### Launch 10 Training Run [Optional]

If you would like to prepare for 10 run submisision, use:

```bash
export NEXP=10
bash run_with_docker.sh
```

After completion, the logs will be available under the directory `$LOGDIR`.

Note:To optimize the machine's performance, the training script will also execute `runtime_tunables.sh` script before any training run.

# 4. Check Quality

### Quality Metric

Cross entropy loss

### Quality Target

0.925
