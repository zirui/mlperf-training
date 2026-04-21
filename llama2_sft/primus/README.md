# LLama2 70B LoRA SFT — Data & Model Preparation (Primus)

This guide covers how to prepare **all** data, model weights, and checkpoints
**before** launching training so that nothing is downloaded on-the-fly.

## Prerequisites

- Access to the `meta-llama/Llama-2-70b-hf` model on HuggingFace
  (requires signing the LLAMA 2 COMMUNITY LICENSE AGREEMENT)
- A HuggingFace access token (`HF_TOKEN`)
- ~270 GB free disk space for model download + conversion
- Docker with the Primus image (or the dev image)

## Quick Reference — Final Directory Layout

After completing all steps, your data directory should look like this:

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
└── huggingface/                       # HF cache (can be removed after conversion)
    └── hub/
        └── models--meta-llama--Llama-2-70b-hf/
```

---

## Step 1: Build the Docker Image

Build the Primus submission image:

```bash
cd llama2_sft/primus
docker build -t rocm/amd-mlperf:llama2_70b_sft_primus_6.0 .
```

Or build the dev image (includes profiling tools):

```bash
cd llama2_sft/primus
bash dev/build_docker.sh
```

## Step 2: Start a Preparation Container

Start an interactive container with the host data directory mounted at `/data`.
This is where all dataset and checkpoint files will be stored.

```bash
export DATADIR=/data/mlperf_llama2/data
sudo mkdir -p $DATADIR

docker run -it --rm \
    -v $DATADIR:/data \
    --net=host --uts=host --ipc=host \
    --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    -e HF_TOKEN=${HF_TOKEN} \
    rocm/amd-mlperf:llama2_70b_sft_primus_6.0 \
    bash
```

All remaining steps run **inside this container**.

## Step 3: Download the Dataset

The dataset is the GovReport dataset, pre-tokenized and packed so that each
sequence has length 8192.

**Option A — Download pre-packed `.npy` files from the AMD HF repo (recommended):**

```bash
pip install huggingface_hub
huggingface-cli login --token $HF_TOKEN

huggingface-cli download \
    amd/mlperf-training-llama2-70b-lorasft-nemo \
    --repo-type=dataset \
    --local-dir /data
```

Verify the files:

```bash
ls -lh /data/train.npy /data/validation.npy
```

**Option B — Download raw parquet and convert (using bundled scripts):**

If Option A is unavailable, use the scripts bundled in the container at
`/workspace/code/scripts/`:

```bash
cd /workspace/code/scripts

# Download raw parquet files from HuggingFace
python download_dataset.py --data_dir /data

# Convert parquet to packed .npy format
python convert_dataset.py --data_dir /data
```

This downloads from `regisss/scrolls_gov_report_preprocessed_mlperf_2`,
converts to `train.npy` / `validation.npy`, and removes the parquet files.

## Step 4: Generate Packed Metadata

Megatron-Bridge requires a `packed_metadata.jsonl` file that describes the
packed sequence properties. Create it:

```bash
cd /workspace/code/scripts
python3 create_metadata.py 8192 /data/packed_metadata.jsonl
```

Or create it manually:

```bash
cat > /data/packed_metadata.jsonl << 'METAEOF'
[
  {
    "max_samples_per_bin": 1,
    "dataset_max_seqlen": 8192,
    "min_packed_seqlen": 8192
  }
]
METAEOF
```

Verify:

```bash
cat /data/packed_metadata.jsonl
```

## Step 5: Download HuggingFace Model Weights

Download the `Llama-2-70b-hf` safetensors to the local HF cache.
This downloads ~130 GB of model weights.

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'meta-llama/Llama-2-70b-hf',
    max_workers=16,
    allow_patterns=['*.safetensors', '*.json', 'tokenizer.model'],
)
print('Download complete.')
"
```

The HF cache defaults to `$HF_HOME/hub` (typically `/data/huggingface/hub`
inside the container when the Primus hook sets `HF_HOME`). You can override
the cache location:

```bash
export HF_HOME=/data/huggingface
```

Verify the download:

```bash
ls $HF_HOME/hub/models--meta-llama--Llama-2-70b-hf/snapshots/*/
# Expected: model-00001-of-00015.safetensors ... model-00015-of-00015.safetensors
#           config.json, tokenizer.model, tokenizer_config.json, etc.
```

## Step 6: Convert to Megatron-Bridge Checkpoint Format

The hook `01_convert_checkpoints.sh` normally runs this conversion
automatically on the first training launch, but running it ahead of time
avoids the on-the-fly overhead.

The conversion script has known import incompatibilities in this container
(`modelopt` broken on ROCm, Megatron-Bridge eagerly importing unsupported
model bridges). The wrapper script handles all of this automatically:

```bash
bash /workspace/code/scripts/convert_megatron_checkpoint.sh \
    "meta-llama/Llama-2-70b-hf" \
    "/data/megatron_checkpoints/Llama-2-70b-hf"
```

This reads from the HF cache (no re-download) and writes the Megatron-format
checkpoint to `/data/megatron_checkpoints/Llama-2-70b-hf/iter_0000000/`.

Verify the conversion:

```bash
ls -lh /data/megatron_checkpoints/Llama-2-70b-hf/iter_0000000/
# Expected:
#   __0_0.distcp           (~64 GB)
#   __0_1.distcp           (~64 GB)
#   common.pt, .metadata, metadata.json
#   run_config.yaml, modelopt_run_config.yaml
#   train_state.pt
#   tokenizer/  (tokenizer.model, tokenizer_config.json, special_tokens_map.json)
```

## Step 7: Final Verification

Run a final check that everything is in place:

```bash
echo "=== Dataset ==="
ls -lh /data/train.npy /data/validation.npy /data/packed_metadata.jsonl

echo ""
echo "=== Megatron Checkpoint ==="
ls -lh /data/megatron_checkpoints/Llama-2-70b-hf/iter_0000000/__0_0.distcp
ls -lh /data/megatron_checkpoints/Llama-2-70b-hf/iter_0000000/__0_1.distcp
ls    /data/megatron_checkpoints/Llama-2-70b-hf/iter_0000000/tokenizer/

echo ""
echo "All data prepared. Ready for training."
```

## Step 8: Exit Preparation Container

```bash
exit
```

---

## (Optional) Clean Up HF Cache

After conversion, the HF cache is no longer needed for training. You can
reclaim ~130 GB:

```bash
sudo rm -rf $DATADIR/huggingface
```

---

## Running Training

With all data prepared, training will start without any downloads:

```bash
export DATADIR=/data/mlperf_llama2/data
export LOGDIR=/data/experiments
export CONT=rocm/amd-mlperf:llama2_70b_sft_primus_6.0

source config_MI355X_1x8x1.sh
export NEXP=1
bash run_with_docker.sh
```
