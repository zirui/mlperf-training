# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pin known-good hipBLASLt/rocBLAS solutions for the HSTU projection GEMMs.

TunableOp keys on M, the jagged token count, which changes every step, so a
tuning file from one step does not hit the next. This synthesizes the cache
line for the current token count and merges it via ``tunable.read_file``.

A pinned index is specific to the library build in EXPECTED_ENV. On a mismatch
the pins do not install and the default heuristic runs.

Enable with GR_PIN_BF16_BLAS_SOLUTIONS=1, which also requires
PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_TUNING=0. To enable only some call sites, pass a comma-separated list of the short
keys in _PINS instead of 1, e.g.
GR_PIN_BF16_BLAS_SOLUTIONS=uvqk_dgrad,out_dgrad.

BF16 is in the name because it is a real limit, not a label. Every key in _PINS
is a BFloat16 TunableOp signature, so a pin fires only when that call site runs
in bf16. With the FP8 forward on, FUSE_QKV and FUSE_OUTLN turn the uvqk and
output projections into torch._scaled_mm calls that no key here can match.

Note on when a pin takes effect: TunableOp fixes an operator's solution the
first time it resolves a given problem key, so a pin is only honoured if it is
installed BEFORE the first GEMM at that token count. That is why the hook sits
ahead of the projection rather than anywhere convenient.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Set, Tuple

import torch

logger: logging.Logger = logging.getLogger(__name__)

# The solution indices below were measured on exactly this stack. They are
# indices into that build's solution registry and are meaningless elsewhere.
EXPECTED_ENV: Dict[str, str] = {
    "PT_VERSION": "2.12.0",
    "HIP_VERSION": "702",
    "HIPBLASLT_VERSION": "100202-dabb6df2b9",
    "GCN_ARCH_NAME": "gfx950:sramecc+:xnack-",
    "ROCBLAS_VERSION": "5.2.0.dabb6df2b9",
}

# (call site, TunableOp op signature, key template on {M}, solution)
#
# Gemm_Rocblas_439683 computes the wrong answer for uvqk_dgrad. It is not in
# this table. Pins are not bit-identical to the heuristic default.
_PINS: List[Tuple[str, str, str, str]] = [
    ("uvqk_fwd", "GemmAndBiasTunableOp_BFloat16_NN",
     "nn_2048_{M}_512_ld_2048_512_2048", "Gemm_Hipblaslt_437615"),
    ("uvqk_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_2048_512_{M}_ld_2048_512_2048", "Gemm_Hipblaslt_436744"),
    ("uvqk_dgrad", "GemmTunableOp_BFloat16_TN",
     "tn_512_{M}_2048_ld_2048_2048_512", "Gemm_Hipblaslt_441461"),
    ("out_fwd", "GemmAndBiasTunableOp_BFloat16_NN",
     "nn_512_{M}_1536_ld_512_1536_512", "Gemm_Hipblaslt_437512"),
    ("out_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_512_1536_{M}_ld_512_1536_512", "Gemm_Hipblaslt_437445"),
    ("out_dgrad", "GemmTunableOp_BFloat16_TN",
     "tn_1536_{M}_512_ld_512_512_1536", "Gemm_Hipblaslt_441460"),
    # Last-layer V/K pack is 1024 wide, so this is a separate key.
    ("targets_vk_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_1024_512_{M}_ld_1024_512_1024", "Gemm_Hipblaslt_436744"),
    # Item-embedding MLP dgrad. Reachable only because _item_forward calls
    # install_for_tokens(all_embeddings.shape[0]) before the MLP.
    ("mlp_b_dgrad", "GemmTunableOp_BFloat16_NN",
     "nn_512_{M}_256_ld_512_256_512", "Gemm_Rocblas_437980"),
    # ContextualPreprocessor MLPs. preproc_dgrad_n512 shares its key with
    # mlp_b_dgrad, so _decide() refuses to activate both.
    ("preproc_dgrad_n1024", "GemmTunableOp_BFloat16_NN",
     "nn_1024_{M}_256_ld_1024_256_1024", "Gemm_Rocblas_437863"),
    ("preproc_dgrad_n512", "GemmTunableOp_BFloat16_NN",
     "nn_512_{M}_256_ld_512_256_512", "Gemm_Rocblas_437491"),
    ("preproc_l2_fwd", "GemmAndBiasTunableOp_BFloat16_TN",
     "tn_512_{M}_256_ld_256_256_512", "Gemm_Hipblaslt_440245"),
    ("preproc_wgrad_n1024", "GemmTunableOp_BFloat16_NT",
     "nt_1024_256_{M}_ld_1024_256_1024", "Gemm_Hipblaslt_436837"),
    ("preproc_l2_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_256_512_{M}_ld_256_512_256", "Gemm_Hipblaslt_436551"),
    ("preproc_wgrad_n512", "GemmTunableOp_BFloat16_NT",
     "nt_512_256_{M}_ld_512_256_512", "Gemm_Hipblaslt_437446"),
    ("preproc_fwd_k24", "GemmAndBiasTunableOp_BFloat16_TN",
     "tn_256_{M}_24_ld_24_24_256", "Gemm_Hipblaslt_441235"),
    ("preproc_wgrad_k24", "GemmTunableOp_BFloat16_NT",
     "nt_24_256_{M}_ld_24_256_24", "Gemm_Hipblaslt_436761"),
]

_active: List[Tuple[str, str, str, str]] = []

_state: Optional[bool] = None  # None = not yet decided, False = disabled
_seen: Set[int] = set()
_path: Optional[str] = None


ENV_VAR: str = "GR_PIN_BF16_BLAS_SOLUTIONS"


def resolve_pin_request() -> str:
    """The requested pin set. Unset or \"0\" means off."""
    return os.environ.get(ENV_VAR, "0")


def _decide() -> bool:
    """Decide once whether pinning is safe and requested, and which pins."""
    want = resolve_pin_request()
    if want in ("0", ""):
        return False

    global _active
    if want in ("1", "all"):
        # "all" keeps the FIRST pin per (signature, key): mlp_b_dgrad and
        # preproc_dgrad_n512 share one key, and the check below refuses both.
        _first: Dict[Tuple[str, str], str] = {}
        _active = [p for p in _PINS
                   if _first.setdefault((p[1], p[2]), p[0]) == p[0]]
    else:
        keys = {k.strip() for k in want.split(",") if k.strip()}
        unknown = keys - {p[0] for p in _PINS}
        if unknown:
            raise ValueError(
                f"{ENV_VAR}={want!r} names unknown pin key(s) "
                f"{', '.join(sorted(unknown))}. Valid keys: "
                f"{', '.join(p[0] for p in _PINS)}. "
                "Refusing to run: an unknown key would silently disable ALL "
                "pins (~4% of step) rather than just the one named."
            )
        _active = [p for p in _PINS if p[0] in keys]
    # Two pins on one (signature, key) would race: TunableOp keeps the FIRST line it
    # reads for a key, so the second solution would be silently ignored.
    seen_keys: Dict[Tuple[str, str], str] = {}
    for name, sig, tmpl, _sol in _active:
        other = seen_keys.setdefault((sig, tmpl), name)
        if other != name:
            raise ValueError(
                f"{ENV_VAR}: pins {other!r} and {name!r} share the key "
                f"{sig}({tmpl}); enable only one of them."
            )
    # DELIBERATELY SOFT, unlike the three checks that raise below. This one and
    # the EXPECTED_ENV validator check are ENVIRONMENT differences, not config
    # errors: the same gin default has to be importable on a non-ROCm box and on
    # a different library build, where the right behaviour is the documented
    # no-op ("on any mismatch this is a no-op and the default heuristic runs").
    # Do not convert these two to raises.
    if not torch.cuda.is_available() or not torch.version.hip:
        logger.warning("BLAS solution pins requested but this is not a ROCm build")
        return False

    import torch.cuda.tunable as tunable

    # Both of the next two are "you asked for pins and will not get them", the
    # same class as an unknown key, so they raise rather than warn. Note
    # apply_env_bootstrap already setdefault()s both of these whenever pins are
    # requested, so reaching either branch means a caller overrode them
    # explicitly -- a config error, not an environment difference.
    if not tunable.is_enabled():
        raise RuntimeError(
            f"{ENV_VAR}={want!r} requested but TunableOp is OFF, so every pin "
            "would be silently ignored. Set PYTORCH_TUNABLEOP_ENABLED=1 (or "
            f"{ENV_VAR}=0 to disable pinning deliberately)."
        )
    if tunable.tuning_is_enabled():
        raise RuntimeError(
            f"{ENV_VAR}={want!r} requested with TunableOp tuning ON. The sweep "
            "overrides the pins and costs minutes per new token count -- and M "
            "changes every step here. Set PYTORCH_TUNABLEOP_TUNING=0."
        )

    live = dict(tunable.get_validators())
    bad = {k: (v, live.get(k)) for k, v in EXPECTED_ENV.items() if live.get(k) != v}
    if bad:
        logger.warning(
            "BLAS solution pins disabled: solution indices were measured on a "
            "different stack, so applying them here could select a different "
            "kernel. Mismatched: %s. Re-tune and update EXPECTED_ENV to enable.",
            ", ".join(f"{k} expected {e!r} got {g!r}" for k, (e, g) in bad.items()),
        )
        return False

    global _path
    _path = os.path.join(tempfile.mkdtemp(prefix="gr_blas_pins_"), "pins.csv")
    logger.info(
        "BLAS solution pins active for %d/%d HSTU projection GEMMs: %s",
        len(_active), len(_PINS), ", ".join(p[0] for p in _active),
    )
    return True


def install_for_tokens(num_tokens: int) -> None:
    """Make the pinned solutions available for this batch's token count.

    Safe to call on every layer of every step: the work happens once per token
    count and is skipped otherwise.
    """
    global _state
    if _state is None:
        _state = _decide()
    if not _state or num_tokens in _seen:
        return
    _seen.add(num_tokens)

    import torch.cuda.tunable as tunable

    lines = [f"Validator,{k},{v}" for k, v in EXPECTED_ENV.items()]
    lines += [
        f"{sig},{tmpl.format(M=num_tokens)},{sol},0.0"
        for _, sig, tmpl, sol in _active
    ]
    assert _path is not None
    try:
        with open(_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        tunable.read_file(_path)
        # GR_PIN_BF16_BLAS_DEBUG=1 reports which token counts were actually
        # registered. "N/8 pins active" is printed once by _decide() and proves
        # only that keys were REQUESTED; it says nothing about the M they were
        # synthesised at. A pin whose GEMM runs at an M that never reaches this
        # function is a silent no-op -- that is exactly how mlp_b_dgrad shipped
        # "active on 8/8 ranks" while running the default kernel. One line per
        # distinct count (so ~1-2 per step), which is why it is off by default.
        if os.environ.get("GR_PIN_BF16_BLAS_DEBUG") == "1":
            logger.info("BLAS pins installed at M=%d for %d key(s): %s",
                        num_tokens, len(_active),
                        ", ".join(p[0] for p in _active))
    except Exception:
        # A failed merge just means the default heuristic runs; never take the
        # training job down over a performance hint.
        logger.warning("failed to install BLAS pins for %d tokens", num_tokens,
                       exc_info=True)
