"""Validate the pinned PyTorch runtime and selective GEMM patch."""

import torch
import torch._inductor.kernel.mm as mm

assert torch.__version__ == "2.12.0+rocm10.0.0", torch.__version__
assert torch.version.hip == "7.15.26333", torch.version.hip
assert mm._PRIMUS_FLUX_SELECTIVE_TRITON
