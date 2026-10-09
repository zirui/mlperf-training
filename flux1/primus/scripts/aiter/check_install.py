"""Validate the selectable AITER atomic attention path."""

from pathlib import Path

assert (
    "PRIMUS_FLUX_AITER_ATOMIC_FP32"
    in Path("/opt/venv/lib/python3.12/site-packages/aiter/ops/mha.py").read_text()
)
