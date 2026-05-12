#!/bin/bash
# Fail (exit 1) if any GPU on this node reports non-zero VRAM%.

set -uo pipefail

HOST="$(hostname)"
smi_output="$(rocm-smi)"

busy=$(printf '%s\n' "$smi_output" \
    | awk '
        /^=+$/      { in_table = !in_table; next }
        !in_table   { next }
        /^[0-9]+[[:space:]]/ {
            vram = $(NF-1)
            if (vram != "0%") print
        }
    ')

if [[ -z "$busy" ]]; then
    echo "[$HOST] check_gpus_idle: OK"
    exit 0
fi

echo "[$HOST] GPU IDLE CHECK FAILED" >&2
echo "$smi_output" >&2
exit 1
