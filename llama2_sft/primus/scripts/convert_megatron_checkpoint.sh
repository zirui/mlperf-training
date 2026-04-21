#!/bin/bash
###############################################################################
# Convert a HuggingFace model to Megatron-Bridge checkpoint format.
#
# Works around two import incompatibilities in the ROCm container:
#   1. modelopt broken on torch.onnx._type_utils
#   2. Megatron-Bridge eagerly importing unsupported model bridges
#
# Usage:
#   bash convert_megatron_checkpoint.sh <hf-model> <megatron-path>
#
# Example:
#   bash convert_megatron_checkpoint.sh \
#       meta-llama/Llama-2-70b-hf \
#       /data/megatron_checkpoints/Llama-2-70b-hf
###############################################################################
set -euo pipefail

HF_MODEL="${1:?Usage: $0 <hf-model> <megatron-path>}"
MEGATRON_PATH="${2:?Usage: $0 <hf-model> <megatron-path>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIMUS_ROOT="${PRIMUS_ROOT:-/workspace/Primus}"

export PYTHONPATH="${PRIMUS_ROOT}/third_party/Megatron-Bridge/src:${PRIMUS_ROOT}/third_party/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"

# Locate modelopt_stub.py
STUB_DIR=""
for d in "$SCRIPT_DIR" \
         "${PRIMUS_ROOT}/scripts" \
         "/workspace/code/scripts"; do
    [[ -f "$d/modelopt_stub.py" ]] && STUB_DIR="$d" && break
done

if [[ -z "$STUB_DIR" ]]; then
    echo "ERROR: modelopt_stub.py not found" >&2
    exit 1
fi

export STUB_DIR HF_MODEL MEGATRON_PATH

# Temporarily disable the broken modelopt package
MODELOPT_DIR="$(python3 -c 'import modelopt, os; print(os.path.dirname(modelopt.__file__))' 2>/dev/null || echo "")"
if [[ -n "$MODELOPT_DIR" && -d "$MODELOPT_DIR" ]]; then
    mv "$MODELOPT_DIR" "${MODELOPT_DIR}_disabled"
    trap 'mv "${MODELOPT_DIR}_disabled" "$MODELOPT_DIR" 2>/dev/null || true' EXIT
fi

cd "$PRIMUS_ROOT"

python3 -c "
import sys, os
sys.path.insert(0, os.environ['STUB_DIR'])
import modelopt_stub
sys.argv = ['convert_checkpoints.py', 'import',
            '--hf-model', os.environ['HF_MODEL'],
            '--megatron-path', os.environ['MEGATRON_PATH']]
exec(open('third_party/Megatron-Bridge/examples/conversion/convert_checkpoints.py').read())
" 
