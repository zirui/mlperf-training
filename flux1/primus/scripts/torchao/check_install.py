"""Validate the TorchAO cast reuse and weight-gradient dispatch changes."""

from pathlib import Path

float8_linear = Path(
    "/opt/venv/lib/python3.12/site-packages/torchao/float8/float8_linear.py"
).read_text()
assert "grad_output_reshaped.shape == (16384, 3072)" in float8_linear
assert "grad_output_reshaped.shape == (32768, 3072)" in float8_linear
assert "PRIMUS_FLUX_REUSE_FP8_INPUT" in float8_linear
assert "((8192, 12288), (8192, 3072))" in float8_linear
