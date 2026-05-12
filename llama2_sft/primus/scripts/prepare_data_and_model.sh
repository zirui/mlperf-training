#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

cd /workspace/code/

# Download and convert dataset
python ${SCRIPT_DIR}/download_dataset.py --data_dir /data
python ${SCRIPT_DIR}/convert_dataset.py --data_dir /data

# Generate packed sequence metadata for Megatron-Bridge
python3 ${SCRIPT_DIR}/create_metadata.py 8192 /data/packed_metadata.jsonl

# Download HF model weights and convert to Megatron-Bridge checkpoint format
bash ${SCRIPT_DIR}/convert_megatron_checkpoint.sh \
    "meta-llama/Llama-2-70b-hf" \
    "/data/megatron_checkpoints/Llama-2-70b-hf"

echo "Data and model preparation completed successfully."
