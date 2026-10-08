# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pin known-good hipBLASLt/rocBLAS solutions for the HSTU projection GEMMs.

hipBLASLt's default heuristic leaves a lot on the table for the projection
shapes: sweeping every registered solution, with a numerical check on, finds
1.06x-1.44x per call site across the five that shipped first, worth ~17
ms/step/rank, and a later sweep of the targets-only layer added a sixth at
1.35x-1.63x. Three of the six are bit-identical to the heuristic pick and three
differ by one to two bf16 ULPs; see the per-pin notes in _PINS. The obstacle to
shipping any of it is the cache key. TunableOp keys on the full problem
descriptor, which includes M, the total jagged token count -- and M changes
every step (2229337, 2376002, 2284350, 2365056 across four consecutive traced
steps), so a tuning file collected at one step never hits at the next.

Everything else in the key is M-independent, and the winning solutions proved
stable across token counts, so rather than tuning per step this synthesizes the
cache line for whichever token count is current and merges it via
``tunable.read_file``. Measured cost is ~0.3 ms of CPU per new token count,
which the GPU covers.

A pinned index only means anything for the library build it was measured on, so
the pins refuse to install unless the TunableOp validators match EXPECTED_ENV
exactly. On any mismatch this is a no-op and the default heuristic runs, which
is merely the current behaviour.

Enable with GR_PIN_BF16_BLAS_SOLUTIONS=1, which also requires
PYTORCH_TUNABLEOP_ENABLED=1 and PYTORCH_TUNABLEOP_TUNING=0. To enable only some
call sites -- for bisecting which pin is responsible for a numerical problem --
pass a comma-separated list of the short keys in _PINS instead of 1, e.g.
GR_PIN_BF16_BLAS_SOLUTIONS=uvqk_dgrad,out_dgrad.

BF16 is in the name because it is a real limit, not a label. Every key in _PINS
is a BFloat16 TunableOp signature, so a pin fires only when that call site runs
in bf16 -- and with the FP8 forward arm on, FUSE_QKV and FUSE_OUTLN turn the
uvqk and output projections into torch._scaled_mm calls that no key here can
match. Four of the six still fire under FP8 (the backward stays bf16 while
DLRM_HSTU_FP8_BWD=0, and three fire when it is 1); uvqk_fwd and out_fwd do not.
Measured: the pins are worth
-3.66% against a bf16 control and -2.86% against an FP8 one.

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
# These come from a sweep run with PYTORCH_TUNABLEOP_NUMERICAL_CHECK=1e-2_1e-2,
# so every solution here was checked against a reference during tuning and no
# candidate could win on speed alone. That matters: the first version of this
# table came from an unchecked sweep, and the solution it picked for uvqk_dgrad
# (Gemm_Rocblas_439683) computes the wrong answer for that shape. Pinning it
# alone drove eval NE to NaN within one eval, while the other five left the
# training curve bit-for-bit unchanged. Both rocBLAS picks are now gone; every
# entry is hipBLASLt.
#
# These are NOT bit-identical to the heuristic default, and cannot be: tuning
# the same shapes at 1e-30 tolerance returns 'Default' for uvqk_dgrad, i.e. no
# alternative solution reproduces it exactly. Expect differences of one to two
# bf16 ULPs, which is ordinary reassociation for this dtype.
#
# Re-run the sweep (scripts alongside this file) whenever EXPECTED_ENV changes.
#
# The MLP forward GEMM is deliberately absent: tuning picked a solution 1.7x
# SLOWER than the default there, so it keeps the heuristic pick.
_PINS: List[Tuple[str, str, str, str]] = [
    # 437878 is what the checked sweep picked here, but it is a tuning-time
    # measurement error: re-measured against a reference at six token counts it
    # runs 9.82/6.28/9.36/8.31 ms where the default runs 8.36/8.41/7.93/7.02,
    # i.e. mostly slower than doing nothing, against the 4.68 ms tuning claimed.
    # 437615 beats the default at every size (6.12/6.24/5.83/5.24) and produces
    # output identical to 437878's to the last digit at every size, so it is not
    # trading accuracy for speed -- same math, better schedule.
    ("uvqk_fwd", "GemmAndBiasTunableOp_BFloat16_NN",
     "nn_2048_{M}_512_ld_2048_512_2048", "Gemm_Hipblaslt_437615"),
    ("uvqk_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_2048_512_{M}_ld_2048_512_2048", "Gemm_Hipblaslt_436744"),
    # Kept for completeness but not worth enabling: 5.29 ms against a 5.33 ms
    # default is a wash, and it is the shape whose previous pin trained to NaN.
    ("uvqk_dgrad", "GemmTunableOp_BFloat16_TN",
     "tn_512_{M}_2048_ld_2048_2048_512", "Gemm_Hipblaslt_441461"),
    ("out_fwd", "GemmAndBiasTunableOp_BFloat16_NN",
     "nn_512_{M}_1536_ld_512_1536_512", "Gemm_Hipblaslt_437512"),
    ("out_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_512_1536_{M}_ld_512_1536_512", "Gemm_Hipblaslt_437445"),
    ("out_dgrad", "GemmTunableOp_BFloat16_TN",
     "tn_1536_{M}_512_ld_512_512_1536", "Gemm_Hipblaslt_441460"),
    # The targets-only last layer's full-row V/K projection backward
    # (## hstu_targets_full_vk_backward ##). Its N is the 1024-wide V/K pack
    # rather than uvqk's 2048, so it is a key of its own and was not in the
    # original sweep. It wants the same solution the uvqk wgrad does and, like
    # that one, is bit-identical to the default at every production token count
    # -- both run the same GSU16 split, so the reduction order does not move.
    # 1.35x-1.63x over the 1.99M-2.38M token counts the step-52 trace contains,
    # worth 1.0-1.4 ms/step/rank -- an offline figure, not an e2e one, and the
    # spread is the gap between scaling the trace's in-situ 4.05 ms by the
    # measured ratio and scaling by the measured absolute delta. The targets-only
    # forward does not call install_for_tokens; it does not have to, because it
    # is the last layer and the ordinary ones ahead of it register the same
    # count -- the traced M sets are identical. A targets-only-only model would
    # miss this pin silently.
    #
    # Small token counts are mixed rather than uniformly worse: measured at
    # 16K it is 1.5x FASTER (0.034 vs 0.052 ms), at 64K a wash (0.082 vs
    # 0.081), and at 1K it is 8.6x SLOWER (0.080 vs 0.009 ms). Every one of
    # those is under 0.08 ms absolute and two orders of magnitude below the
    # counts a yambda-5b step produces, but it is why this entry claims
    # "production token count" and not "any" -- the solution is tuned for a
    # K=M reduction long enough to amortise its non-temporal schedule.
    #
    # There is no matching dgrad entry. The checked sweep picked
    # Gemm_Hipblaslt_441462 and claimed 1.73 ms, but re-measured it runs
    # 3.57 ms against a 3.32 ms default over twelve paired points -- the same
    # tuning-time measurement error 437878 had on uvqk_fwd. The runner-up,
    # Gemm_Hipblaslt_441024, launches a kernel whose name is identical to the
    # default's, i.e. it is the heuristic pick under another index.
    ("targets_vk_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_1024_512_{M}_ld_1024_512_1024", "Gemm_Hipblaslt_436744"),
    # The item-embedding MLP's second Linear (512 -> 256), DATA-GRADIENT side:
    # grad_in[M,512] = grad_out[M,256] @ w[256,512]. Not an HSTU projection --
    # it is in DlrmHSTU._item_embedding_mlp -- but it is the largest single
    # small-K bf16 GEMM in the step.
    #
    # REACHABILITY IS NOT FREE HERE, and an earlier version of this comment got
    # it wrong. It claimed the dgrad "runs in the backward, after the install,
    # so it is in time". Ordering was never the problem -- the KEY was. The MLP
    # row count is not the HSTU token count: measured on 8 ranks x 5 steps of a
    # profiled run, num_tokens exceeds all_embeddings.shape[0] by exactly 8192
    # (== global_batch_size) at every one of 40 points, because preprocess()
    # appends one token per sample between the MLP and the projection. So the
    # line install_for_tokens() wrote was keyed on an M this GEMM never asks
    # about: 0/40 lookups hit, and TunableOp fell through to the heuristic.
    #
    # This was caught only by a trace, and only because the trace was taken. The
    # pin reported "active on 8/8 ranks" in every driver log and the arm still
    # ran the DEFAULT kernel -- Cijk_..._MT256x256x32_MI32x32x1_... byte for
    # byte identical to the control, 1.5395 vs 1.5875 ms, against a predicted
    # 2.14x. The corresponding 200-step e2e ladder read -0.15% because it was
    # comparing two identical configurations. INSTALLED IS NOT USED, and the
    # "N/8 pins active" line proves only the former.
    #
    # It is reachable now because modules/dlrm_hstu.py:_item_forward calls
    # install_for_tokens(all_embeddings.shape[0]) before the MLP. Remove that
    # call and this entry silently goes inert again.
    #
    # 2.17x, and unusually well-behaved: measured paired against the heuristic
    # at EIGHT token counts spanning 1.98M-2.42M, 3 rounds each (n=24), one
    # process per point. Per-count medians 2.055/2.058/2.084/2.168/2.173/
    # 2.177/2.183/2.208 -- every count wins, and both arms are monotone in M
    # with a within-count spread under 0.04. That across-M stability is not a
    # nicety, it is the precondition this whole file runs on: install_for_tokens
    # synthesises the line for WHATEVER count is current, so a pin fires at
    # every M and a candidate that wins only on a pooled median would be a
    # speedup plus a regression on the steps that land badly.
    #
    # THE SIBLING FORWARD PIN IS DELIBERATELY ABSENT, and not because it is
    # slow: b_fwd screens at 1.06x, which is inside the screen's own noise and
    # not worth a second variable. It is now reachable (the _item_forward
    # install precedes the MLP call), so if it is ever wanted it needs only a
    # measurement, not a plumbing change.
    #
    # The other three small-K candidates were all rejected on measurement:
    #  * d_fwd (multitask head, 1.22x pooled) LOSES at 3 of 8 token counts
    #    (0.872/0.885/0.917). The candidate is smooth and monotone in M; it is
    #    the DEFAULT that is bimodal, dropping to a fast path at those counts.
    #    A static pin gives up that adaptivity, so the pooled median is a
    #    fiction. Pinning is only safe where the default is uniformly mediocre.
    #  * c_fwd's 12.66x screen was an artifact: the pinned arm launches NO
    #    kernel and still "times" 0.0114 ms, implying 2370 TF/s against a
    #    measured 2106 TF/s bf16 peak. Where a kernel really runs it is ~1.05x.
    #  * a_dgrad is a null (2.68-2.81 ms against a 2.05-3.32 ms default):
    #    lower variance, not faster.
    #
    # This is the file's only rocBLAS entry, which the note above says were all
    # removed -- so it gets the scrutiny that history earns. It is NOT the
    # 439683 wrong-answer mode: 99.9928% of elements are bit-identical to the
    # default and the misses are monotone in smallness (elements above 1024 ULP
    # have |ref| 9.3e-10 against a typical 0.0269), max absolute error is one
    # bf16 ULP at the top of range, RMS SNR 94.1 dB. That is ordinary
    # reassociation, not a different answer.
    ("mlp_b_dgrad", "GemmTunableOp_BFloat16_NN",
     "nn_512_{M}_256_ld_512_256_512", "Gemm_Rocblas_437980"),
    # ---- ContextualPreprocessor MLP GEMMs (fused_linear_norm._LinearNormFunction):
    # content (Din=512), additional (Din=1024 = 2 x 512) and action (Din=24) MLPs,
    # each Linear(Din->256), SwishLN(256), Linear(256->512), LN(512). Swept iter29
    # (probes/iter29/gemm): TunableOp exhaustive sweep with a 1e-2 numerics check at
    # M=2334943, then the top 8 re-timed against the heuristic INTERLEAVED on one
    # MI350X (25 reps/arm, 512 MiB cache flush before every call) at six token
    # counts 1.70M-2.42M. Every default-ON entry below wins at all six counts.
    # Bit-identity: the wgrad entry is bit-identical at every count; the other three
    # differ from the heuristic on ~7e-5 of elements by <=2 bf16 ULP (fp32-order
    # differences, identical SNR vs an fp32 reference to 3 decimals).
    #
    # Reachability: these GEMMs run at the preprocessor's row count, which is
    # the item-MLP row count _item_forward already registers; preprocessors.py
    # also registers it itself so the pins do not depend on that ordering.
    #
    # NOTE preproc_dgrad_n512 shares its KEY with mlp_b_dgrad (same dims and layout
    # in another call site). One key can take only one solution, so _decide() refuses
    # to activate both.
    #
    # additional-MLP L1 dgrad dx[M,1024] = dh[M,256] @ w[256,1024]:
    # 4.41 -> 1.93 ms at 2.33M, 2.29x-2.40x over 1.7M-2.42M.
    ("preproc_dgrad_n1024", "GemmTunableOp_BFloat16_NN",
     "nn_1024_{M}_256_ld_1024_256_1024", "Gemm_Rocblas_437863"),
    # content-MLP L1 dgrad dx[M,512] = dh[M,256] @ w[256,512]:
    # 2.19 -> 1.02 ms at 2.33M, 2.04x-2.21x.
    ("preproc_dgrad_n512", "GemmTunableOp_BFloat16_NN",
     "nn_512_{M}_256_ld_512_256_512", "Gemm_Rocblas_437491"),
    # L2 forward h[M,512] = addmm(b, x[M,256], w[512,256].t()), three calls/step:
    # 1.65 -> 1.09 ms at 2.33M, 1.42x-1.62x.
    ("preproc_l2_fwd", "GemmAndBiasTunableOp_BFloat16_TN",
     "tn_512_{M}_256_ld_256_256_512", "Gemm_Hipblaslt_440245"),
    # additional-MLP L1 wgrad dW[256,1024] = dh.t() @ x: BIT-IDENTICAL,
    # 1.95 -> 1.41 ms at 2.33M, 1.26x-1.45x.
    ("preproc_wgrad_n1024", "GemmTunableOp_BFloat16_NT",
     "nt_1024_256_{M}_ld_1024_256_1024", "Gemm_Hipblaslt_436837"),
    # Below the 10% / 0.1 ms-per-call bar; listed, NOT in the gin default.
    # L2 wgrad dW[512,256] (3 calls/step), bit-identical, 1.09x-1.11x (-0.09 ms/call).
    ("preproc_l2_wgrad", "GemmTunableOp_BFloat16_NT",
     "nt_256_512_{M}_ld_256_512_256", "Gemm_Hipblaslt_436551"),
    # content L1 wgrad dW[256,512], bit-identical, 1.02x-1.32x.
    ("preproc_wgrad_n512", "GemmTunableOp_BFloat16_NT",
     "nt_512_256_{M}_ld_512_256_512", "Gemm_Hipblaslt_437446"),
    # action L1 fwd addmm [M,24]x[24,256], bit-identical, 1.16x-1.26x (-0.07 ms).
    ("preproc_fwd_k24", "GemmAndBiasTunableOp_BFloat16_TN",
     "tn_256_{M}_24_ld_24_24_256", "Gemm_Hipblaslt_441235"),
    # action L1 wgrad dW[256,24]: 1.36x-1.53x (-0.09 ms) but NOT bit-identical
    # (1.8e-3 of elements differ by 1 bf16 ULP).
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
            # FAIL LOUD. This used to warn and return False, which turned OFF
            # EVERY pin -- not just the unrecognised one -- and cost ~4% of step
            # while the launch log still said pins were requested. A perf
            # regression with no error anywhere is the worst possible outcome:
            # it reads as a result rather than a misconfiguration. A key naming
            # a pin that does not exist is a config error, so raise.
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
