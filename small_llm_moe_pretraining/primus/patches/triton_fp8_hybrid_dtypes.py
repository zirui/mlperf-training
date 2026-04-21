"""Patch Triton FP8 grouped GEMM backends to accept hybrid (mixed e4m3/e5m2) dtypes.

The Triton JIT kernels are dtype-agnostic: tl.dot handles mixed FP8 natively on
gfx950 via v_mfma_f32_32x32x64_f8f6f4 with per-operand cbsz/blgp modifiers.
This patch aligns the Triton dispatcher dtype gates with hipBLASLt, which already
supports hybrid pairs.

Target: primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py
"""
import glob
import pathlib

candidates = glob.glob("/opt/venv/lib/python*/site-packages/primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py") + \
             glob.glob("/workspace/deps/Primus-Turbo/primus_turbo/pytorch/kernels/grouped_gemm/grouped_gemm_fp8_impl.py")
assert candidates, "grouped_gemm_fp8_impl.py not found"
target = pathlib.Path(candidates[0])
src = target.read_text()

old = "set(_COMMON_SUPPORTED_DTYPES)"
new = "set(_COMMON_SUPPORTED_DTYPES + _HYBRID_SUPPORTED_DTYPES)"

patched = 0
lines = src.split("\n")
out = []
for line in lines:
    if old in line and "Triton" in "".join(out[-20:]):
        out.append(line.replace(old, new))
        patched += 1
    else:
        out.append(line)

if patched == 0:
    if new in src:
        print(f"Already patched {target}: hybrid FP8 dtypes already enabled (idempotent)")
    else:
        raise RuntimeError(f"Expected 2 replacements in Triton backends, got 0 (and target not already patched)")
else:
    assert patched == 2, f"Expected 2 replacements in Triton backends, got {patched}"
    target.write_text("\n".join(out))
    print(f"Patched {target}: {patched} Triton backends now accept hybrid FP8 dtypes")
