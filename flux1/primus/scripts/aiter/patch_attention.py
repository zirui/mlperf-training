"""Keep AITER FP32 atomic attention selectable at runtime."""

from pathlib import Path

path = Path("/opt/venv/lib/python3.12/site-packages/aiter/ops/mha.py")
source = path.read_text().replace(
    "from typing import Any, Optional, Tuple\n",
    "from typing import Any, Optional, Tuple\n\nimport os\n",
    1,
)
old = "        True,  # is_v3_atomic_fp32\n"
new = '        os.getenv("PRIMUS_FLUX_AITER_ATOMIC_FP32", "1") == "1",  # is_v3_atomic_fp32\n'
assert source.count(old) == 1
path.write_text(source.replace(old, new))
