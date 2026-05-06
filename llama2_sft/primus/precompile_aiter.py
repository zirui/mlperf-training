import os

os.environ.setdefault("GPU_ARCHS", "gfx950")

from aiter.jit.core import get_args_of_build, build_module

MODULES = [
    "module_aiter_enum",
    "module_rope_general_fwd",
    "module_rope_general_bwd",
    "module_gemm_a4w4_asm",
]

for name in MODULES:
    args = get_args_of_build(name)
    build_module(
        name,
        args["srcs"],
        args["flags_extra_cc"],
        args["flags_extra_hip"],
        args["blob_gen_cmd"],
        args["extra_include"],
        args["extra_ldflags"],
        args["verbose"],
        args["is_python_module"],
        args["is_standalone"],
        args["torch_exclude"],
        args.get("hipify", False),
    )
    print(f"  compiled {name}")

print("AITER pre-compile OK")
