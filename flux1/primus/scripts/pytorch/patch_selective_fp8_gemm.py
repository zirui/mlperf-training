"""Prefer Triton for the qualified FLUX FP8 GEMM shapes."""

from pathlib import Path

path = Path("/opt/venv/lib/python3.12/site-packages/torch/_inductor/kernel/mm.py")
source = path.read_text()
source = source.replace("import logging\n", "import logging\nimport os\n", 1)
old = """    if use_aten_gemm_kernels():
        templates_to_use.append(aten__fp8_mm)
"""
new = """    selective_triton = os.environ.get("PRIMUS_FLUX_FP8_GEMM_BACKEND") == "selective_triton"
    if use_aten_gemm_kernels() and not (
        selective_triton and (m, n, k) in {(8192, 3072, 12288), (16384, 3072, 15360)}
    ):
        templates_to_use.append(aten__fp8_mm)
"""
assert source.count(old) == 1, "unexpected torch/_inductor/kernel/mm.py"
source = source.replace(old, new)
source += "\n_PRIMUS_FLUX_SELECTIVE_TRITON = True\n"
path.write_text(source)
