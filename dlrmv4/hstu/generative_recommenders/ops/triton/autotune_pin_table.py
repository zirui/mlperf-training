"""Robust per-key Triton autotune pins (``DLRM_HSTU_FP8_AUTOTUNE_PINS``).

Default ON through the gin binding ``apply_env_bootstrap.DLRM_HSTU_FP8_AUTOTUNE_PINS =
True`` (yambda_5b.gin, ship-4 item (10)); absent from the environment = off. Opt out per
run with ``DLRM_HSTU_FP8_AUTOTUNE_PINS=0``. Ported from the Arbor testbed (commit
14006f6, iteration 20) onto b97d9d9; the kernel modules and their autotune config
lists are identical between the two trees.

Why: every cold start benchmarks ~21 multi-config autotune keys inside the MLPerf timed
region (~31 s/run, iter-18 ``startup-attribution``), ranks pick different winners on
near-ties (coin flips, ~0.10 ms/rank-step), and the winner chosen on the early-window
shape is kept for the whole run although the keys do not contain the token count M
(early winners at late shapes cost ~0.51 ms/rank-step, iter-19 ``autotune-pin-table``).

What: seed each Autotuner's in-memory ``cache[key]`` with the config that minimises the
WORST normalised autotune time across four shape regimes (START_TS 0 twice, 150, 280;
8 ranks x 2 hosts). Source: eval/eval_autotune_regret.py over probes/iter19/attime_{a,b};
table proposals/autotune_pin_table_iter19.md. A seeded key skips benchmarking; any key not
in the table (new shape/dtype) is autotuned exactly as before. Only configs that are
already in the kernel's own autotune list are used, so every pinned launch is one the
autotuner could have chosen itself. ``TRITON_FULL_AUTOTUNE=1`` disables the pins.

Numerics: BLOCK_N / num_warps change rows per program. Per-row LayerNorm statistics are
unchanged, but the dw/db partial-sum reduction order of ``*_bwd*`` kernels can differ
from whatever winner a rank would have picked (which already varies run to run).
Declared as a reduction-order change.
"""

import ast
import logging
import os
from typing import Dict, List, Tuple

logger: logging.Logger = logging.getLogger(__name__)

# Grep-able arming literal: a run log with "armed OK: [autotune-pins] seeded 21/21 keys"
# on every rank proves the table is active. Not printed when 0 keys were seeded.
ARMED_PREFIX = "armed OK: [autotune-pins]"

_PKG = "generative_recommenders.ops.triton."

# (module, kernel, str(autotune key), robust config)
PIN_TABLE: List[Tuple[str, str, str, Dict[str, int]]] = [
    ('triton_hstu_linear', '_ln_mul_dropout_bwd_dwdb', "(512, 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.bfloat16')", {'BLOCK_N': 128, 'num_warps': 8, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_hstu_linear', '_ln_mul_dropout_fwd_rng', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.int8')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_hstu_linear', '_ln_mul_dropout_fwd_rng', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.int8')", {'BLOCK_N': 4, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.54%
    ('triton_jagged_tensors', 'concat_2D_jagged_multirow', "(1, 'torch.int64', 'torch.int64', 'torch.int64', 'torch.int64')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_jagged_tensors', 'concat_2D_jagged_multirow', "(1, 'torch.int64', 'torch.int64', 'torch.int64', 'torch.int64', 'torch.int64')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_jagged_tensors', 'concat_2D_jagged_multirow', "(32, 'torch.float32', 'torch.float32', 'torch.int64', 'torch.int64', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_jagged_tensors', 'concat_2D_jagged_multirow', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.int64', 'torch.bfloat16')", {'BLOCK_N': 2, 'num_warps': 1, 'num_stages': 3}),  # worst regret 1.42%
    ('triton_jagged_tensors', 'split_2D_jagged_multirow', "(32, 'torch.float32', 'torch.int64', 'torch.int64', 'torch.float32', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_jagged_tensors', 'split_2D_jagged_multirow', "(512, 'torch.bfloat16', 'torch.int64', 'torch.bfloat16', 'torch.bfloat16')", {'BLOCK_N': 4, 'num_warps': 4, 'num_stages': 3}),  # worst regret 3.25%
    ('triton_layer_norm', '_weighted_layer_norm_bwd_dx', "(256, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_layer_norm', '_weighted_layer_norm_bwd_dx', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # worst regret 1.15%
    ('triton_layer_norm', '_weighted_layer_norm_bwd_dx', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # worst regret 2.84%
    ('triton_layer_norm', '_weighted_layer_norm_bwd_dx', "(512, 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(256, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 4, 'num_stages': 3}),  # worst regret 1.45%
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32')", {'BLOCK_N': 4, 'num_warps': 1, 'num_stages': 3}),  # worst regret 1.72%
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # worst regret 1.86%
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.bfloat16', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.bfloat16')", {'BLOCK_N': 4, 'num_warps': 1, 'num_stages': 3}),  # worst regret 1.14%
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.bfloat16', 'torch.float8_e4m3fn', 'torch.float8_e4m3fn', 'torch.bfloat16', 'torch.bfloat16', 'torch.float32', 'torch.float32')", {'BLOCK_N': 8, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_layer_norm', '_weighted_layer_norm_fwd', "(512, 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32', 'torch.float32')", {'BLOCK_N': 1, 'num_warps': 1, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_position', '_add_embeddings_bwd_kernel', "(4096, 1024, 512, 'torch.bfloat16', 'torch.int32', 'torch.int64', 'torch.float32')", {'BLOCK': 64, 'num_warps': 2, 'num_stages': 3}),  # worst regret 0.00%
    ('triton_position', '_add_timestamp_position_embeddings_kernel', "(4096, 'torch.bfloat16', 'torch.int64', 'torch.int64', 'torch.float32', 'torch.float32', 'torch.bfloat16', 'torch.int64', 'torch.int32', 'torch.int32', 'torch.int64')", {'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 2}),  # worst regret 2.25%
]


def _cfg_matches(cfg, want: Dict[str, int]) -> bool:
    for name, val in want.items():
        if name == "num_warps":
            if cfg.num_warps != val:
                return False
        elif name == "num_stages":
            if cfg.num_stages != val:
                return False
        elif cfg.kwargs.get(name) != val:
            return False
    # every tunable kwarg of the config must be named by the pin
    return set(cfg.kwargs) <= set(want)


def apply_autotune_pins(force: bool = False) -> Tuple[int, int]:
    """Seed the autotune caches. Returns (seeded, total). Safe to call more than once."""
    if not force and os.environ.get("DLRM_HSTU_FP8_AUTOTUNE_PINS", "0") != "1":
        print(
            "[autotune-pins] OFF (DLRM_HSTU_FP8_AUTOTUNE_PINS!=1): autotuning every key",
            flush=True,
        )
        return 0, 0
    if os.environ.get("TRITON_FULL_AUTOTUNE", "0") == "1":
        print("[autotune-pins] OFF: TRITON_FULL_AUTOTUNE=1 disables the pins", flush=True)
        return 0, len(PIN_TABLE)
    import importlib

    seeded = 0
    problems = []
    for mod_name, kern, key_str, want in PIN_TABLE:
        try:
            mod = importlib.import_module(_PKG + mod_name)
            at = getattr(mod, kern)
            # unwrap @triton.heuristics / other decorators layered over the Autotuner
            for _ in range(4):
                if hasattr(at, "cache") and hasattr(at, "configs"):
                    break
                at = at.fn
            cache = at.cache
            configs = at.configs
        except Exception as e:  # pragma: no cover - environment specific
            problems.append(f"{kern}: {type(e).__name__}: {e}")
            continue
        match = [c for c in configs if _cfg_matches(c, want)]
        if len(match) != 1:
            problems.append(f"{kern} {key_str[:40]}: {len(match)} configs match {want}")
            continue
        key = ast.literal_eval(key_str)
        if key in cache and cache[key] is not match[0]:
            problems.append(f"{kern} {key_str[:40]}: already tuned, left as is")
            continue
        cache[key] = match[0]
        seeded += 1
    if seeded > 0:
        msg = f"{ARMED_PREFIX} seeded {seeded}/{len(PIN_TABLE)} keys"
    else:
        msg = f"[autotune-pins] NOT armed: seeded 0/{len(PIN_TABLE)} keys"
    if problems:
        msg += "; skipped: " + " | ".join(problems)
    print(msg, flush=True)
    return seeded, len(PIN_TABLE)
