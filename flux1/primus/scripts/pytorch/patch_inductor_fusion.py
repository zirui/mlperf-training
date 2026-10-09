"""Fall back when an Inductor fusion group cannot be compiled."""

from pathlib import Path

path = Path("/opt/venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py")
source = path.read_text()
old = """        else:
            # Start parallel compilation for all three kernels
            future_and_mod_l1 = self.compile_kernel(node_list_1)
            future_and_mod_l2 = self.compile_kernel(node_list_2)
            future_and_mod_l1_fused = self.compile_kernel(node_list_fused)

            def benchmark_when_ready() -> bool:
"""
new = """        else:
            # Unsupported fusion groups should not abort the whole graph compile.
            try:
                future_and_mod_l1 = self.compile_kernel(node_list_1)
                future_and_mod_l2 = self.compile_kernel(node_list_2)
                future_and_mod_l1_fused = self.compile_kernel(node_list_fused)
            except (NotImplementedError, CompilationError):
                return FusionResult.fuse(False)

            def benchmark_when_ready() -> bool:
"""
assert source.count(old) == 1
path.write_text(source.replace(old, new))
