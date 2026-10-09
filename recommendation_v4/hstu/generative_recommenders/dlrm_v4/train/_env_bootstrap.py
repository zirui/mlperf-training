# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gin-driven env-var bootstrap.

Some env vars must be set *before* certain modules import (e.g. Triton's
`@triton.autotune` decorator reads `TRITON_FULL_AUTOTUNE` at module load
time, well before `gin.parse_config_file` runs in the default ordering).

`apply_env_bootstrap()` is `@gin.configurable`, so the gin file becomes the
canonical source of truth. `train_ranker.py` parses gin with
`skip_unknown=True` early in `_main_func`, calls this function to push the
bindings into `os.environ`, then does the heavy imports.
"""

import logging
import os
from typing import Dict, Optional, Tuple, Union

import gin

logger: logging.Logger = logging.getLogger(__name__)

# The FP8 HSTU forward knob set. It is one unit: a partial set is a different
# configuration. A pre-set environment variable still wins per knob.
_FP8_FORWARD_ARM: Dict[str, str] = {
    # e4m3 operands for the HSTU linear layers, and for attention q/k/v.
    "DLRM_HSTU_FP8_GEMM": "1",
    "DLRM_HSTU_FP8_ATTN": "1",
    # Fold each quantization into the op that already holds the value in
    # registers. Quantizing as its own pass costs more than the gemm saves.
    "DLRM_HSTU_FP8_FUSE_LN": "1",
    "DLRM_HSTU_FP8_FUSE_OUTLN": "1",
    "DLRM_HSTU_FP8_FUSE_QKV": "1",
    "DLRM_HSTU_FP8_FUSE_SILU": "1",
    "DLRM_HSTU_FP8_FUSE_RESIDUAL": "1",
    "DLRM_HSTU_FP8_FUSE_RESIDUAL_ALGORITHM": "1",
    # Keep the e4m3 uvqk pack so the backward re-reads it instead of
    # recomputing the projection. All three HSTU layers. A layer mask selects
    # which layers keep the pack.
    "DLRM_HSTU_FP8_SAVE_UVQK": "1",
    "DLRM_HSTU_FP8_SAVE_UVQK_LAYERS": "0,1,2",
    "DLRM_HSTU_FP8_EARLY_RELEASE_UVQK": "1",
    # Calibrate four steps then freeze. Freezing is not a numerical convenience:
    # it is what removes the per-step amax and the blocking device-to-host scale
    # read from the steady-state step.
    "DLRM_HSTU_FP8_SCALE_MODE": "frozen",
    "DLRM_HSTU_FP8_CALIBRATION_STEPS": "4",
    "DLRM_HSTU_FP8_HOST_SCALE_VALUES": "1",
    "DLRM_HSTU_FP8_GEMM_ASCALE_MARGIN": "1.5",
    # Raise rather than quietly serving bf16 under an FP8 label.
    "DLRM_HSTU_FP8_FAIL_CLOSED": "1",
    # fp8 accumulate trades the accumulator precision this arm converged with.
    "DLRM_HSTU_FP8_GEMM_FASTACC": "0",
}

# The knobs that actually engage FP8 compute. Any of these set without the
# master switch is a partial configuration, so it fails loudly rather than
# degrading silently.
# Spelled out rather than imported from ops/blas_solution_pins.py, which owns
# them: this module exists to set env vars BEFORE the heavy imports, and that
# module imports torch. Keep the two spellings in step by hand -- a bootstrap
# that exports a name the reader does not check is a silent no-op.
_PIN_ENV: str = "GR_PIN_BF16_BLAS_SOLUTIONS"

_FP8_COMPUTE_KNOBS: Tuple[str, ...] = (
    "DLRM_HSTU_FP8_GEMM",
    "DLRM_HSTU_FP8_ATTN",
    "DLRM_HSTU_FP8_FUSE_LN",
    "DLRM_HSTU_FP8_FUSE_OUTLN",
    "DLRM_HSTU_FP8_FUSE_QKV",
    "DLRM_HSTU_FP8_FUSE_SILU",
    "DLRM_HSTU_FP8_FUSE_RESIDUAL",
    "DLRM_HSTU_FP8_SAVE_UVQK",
)



def _bind_bool(name: str, value: Optional[bool]) -> None:
    """Push a gin-bound boolean into the environment; a pre-set env var wins.

    Same env-wins rule as the two bindings written out below, factored out
    because each additional kernel flag needs it identically.
    """
    if name in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set %s=%s (overrides gin binding)",
            name,
            os.environ[name],
        )
    elif value is not None:
        os.environ[name] = "1" if value else "0"
        logger.info("env bootstrap: %s=%s", name, os.environ[name])


def _truthy(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "on", "yes")


# Folded into DLRM_HSTU_FP8_BWD_FULL: block dO scales and
# the saturating cast are now its only code path. Kept as names only so a
# stale opt-out fails loudly instead of being silently ignored.
_REMOVED_C8_KNOBS: Tuple[str, ...] = (
    "DLRM_HSTU_FP8_BWD_FULL_DOBLK",
    "DLRM_HSTU_FP8_BWD_FULL_SAT",
)


def _check_removed_c8_knobs() -> None:
    """Raise if a removed C8 sub-knob is set to an off value.

    Mirrors ops/flydsl/hstu_flydsl_mha.py check_removed_c8_knobs (this module
    must not import torch). An on value is the only behaviour and is accepted.
    """
    for name in _REMOVED_C8_KNOBS:
        raw = os.environ.get(name)
        if raw is not None and raw.strip() != "" and not _truthy(raw):
            raise RuntimeError(
                f"{name}={raw!r}: this knob was removed; DOBLK+SAT is the only "
                "C8 path. Use DLRM_HSTU_FP8_BWD_FULL=0 to turn C8 off."
            )


def _bind_value(name: str, value: Optional[Union[str, int, float]]) -> None:
    """Push a gin-bound non-boolean into the environment; a pre-set env var wins.

    Separate from `_bind_bool` because a bool has to serialize as "1"/"0": the
    readers compare against those literals, not against `str(True)`.
    """
    if name in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set %s=%s (overrides gin binding)",
            name,
            os.environ[name],
        )
    elif value is not None:
        os.environ[name] = str(value)
        logger.info("env bootstrap: %s=%s", name, os.environ[name])


@gin.configurable
def apply_env_bootstrap(
    TRITON_FULL_AUTOTUNE: Optional[bool] = None,
    HSTU_ATTN_BWD: Optional[str] = None,
    HSTU_ATTN_FWD: Optional[str] = None,
    TRITON_HSTU_FAST_INTERIOR: Optional[bool] = None,
    TRITON_HSTU_LAST_LAYER_TARGETS_ONLY: Optional[bool] = None,
    TRITON_HSTU_TARGETS_ONLY_EVAL: Optional[bool] = None,
    GR_PIN_BF16_BLAS_SOLUTIONS: Optional[str] = None,
    DLRM_HSTU_FP8_FWD: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_QK_MFMA: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_FULL: Optional[bool] = None,
    DLRM_HSTU_FP8_MASK16: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_EPILOGUE: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_DQ_SPLIT: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_PIPE: Optional[int] = None,
    DLRM_HSTU_FP8_EC_CAT_FUSE: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_STRIDED_GRAD: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_FAST: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_BWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID: Optional[bool] = None,
    DLRM_HSTU_FP8_LNMD_BWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_SAVE_NORMED_X: Optional[bool] = None,
    DLRM_HSTU_FP8_SAVE_TARGETS_VK: Optional[bool] = None,
    DLRM_HSTU_FP8_COLSUM: Optional[bool] = None,
    DLRM_HSTU_FP8_SYNCFREE: Optional[bool] = None,
    DLRM_HSTU_FP8_GC_FREEZE: Optional[bool] = None,
    DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM: Optional[bool] = None,
    DLRM_HSTU_FP8_PREBUILD: Optional[bool] = None,
    DLRM_HSTU_FP8_AUTOTUNE_PINS: Optional[bool] = None,
    DLRM_HSTU_FP8_ALLOC_PREGROW: Optional[str] = None,
) -> None:
    # A pre-set environment variable wins over the gin binding. The pinned
    # triton configs are MI350X-specific, so a different GPU arch (e.g. B200
    # sm_100) sets TRITON_FULL_AUTOTUNE=1 in the launcher environment to
    # re-enable the full autotune search WITHOUT editing this (AMD-default)
    # gin file. Cross-cluster launchers thus stay config-as-code via env.
    if "TRITON_FULL_AUTOTUNE" in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set TRITON_FULL_AUTOTUNE=%s (overrides gin binding)",
            os.environ["TRITON_FULL_AUTOTUNE"],
        )
    elif TRITON_FULL_AUTOTUNE is not None:
        os.environ["TRITON_FULL_AUTOTUNE"] = "1" if TRITON_FULL_AUTOTUNE else "0"
        logger.info("env bootstrap: TRITON_FULL_AUTOTUNE=%s", os.environ["TRITON_FULL_AUTOTUNE"])

    # Same env-wins rule, for the same reason: the FlyDSL backward is gfx950-only
    # and is the default, so a non-gfx950 launcher sets HSTU_ATTN_BWD=triton in
    # the environment rather than forking this (AMD-default) gin file. Without
    # that override an unsupported host raises instead of silently running the
    # slower backward -- see ops/hstu_attention.py.
    if "HSTU_ATTN_BWD" in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set HSTU_ATTN_BWD=%s (overrides gin binding)",
            os.environ["HSTU_ATTN_BWD"],
        )
    elif HSTU_ATTN_BWD is not None:
        os.environ["HSTU_ATTN_BWD"] = HSTU_ATTN_BWD
        logger.info("env bootstrap: HSTU_ATTN_BWD=%s", os.environ["HSTU_ATTN_BWD"])

    # The forward half of the same choice, selected separately because the two
    # kernels are decoupled (HSTU has no softmax, so the forward hands the
    # backward nothing). FlyDSL is the gin default (e4m3 forward + fused-silu
    # when DLRM_HSTU_FP8_FWD is on). HSTU_ATTN_FWD=triton is the opt-out to
    # the Triton e4m3 forward. FlyDSL backward stays the default.
    if "HSTU_ATTN_FWD" in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set HSTU_ATTN_FWD=%s (overrides gin binding)",
            os.environ["HSTU_ATTN_FWD"],
        )
    elif HSTU_ATTN_FWD is not None:
        os.environ["HSTU_ATTN_FWD"] = HSTU_ATTN_FWD
        logger.info("env bootstrap: HSTU_ATTN_FWD=%s", os.environ["HSTU_ATTN_FWD"])

    # Both are read at Triton/module import time, which is why they belong here
    # rather than in a config object the model reads at construction.
    _bind_bool("TRITON_HSTU_FAST_INTERIOR", TRITON_HSTU_FAST_INTERIOR)
    _bind_bool(
        "TRITON_HSTU_LAST_LAYER_TARGETS_ONLY", TRITON_HSTU_LAST_LAYER_TARGETS_ONLY
    )
    _bind_bool("TRITON_HSTU_TARGETS_ONLY_EVAL", TRITON_HSTU_TARGETS_ONLY_EVAL)

    # Which HSTU projection GEMMs get a pinned hipBLASLt solution: "0" for none,
    # "1" for every pin in the table, or a comma-separated list of pin names
    # (see ops/blas_solution_pins.py). Env wins, as above.
    #
    # BF16 is in the name because the pins are BF16-only and that is not
    # obvious at the call site: with the FP8 forward arm on, two of the five
    # name projections that have become torch._scaled_mm and stop applying.
    if _PIN_ENV in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set %s=%s (overrides gin binding)",
            _PIN_ENV, os.environ[_PIN_ENV],
        )
    elif GR_PIN_BF16_BLAS_SOLUTIONS is not None:
        os.environ[_PIN_ENV] = GR_PIN_BF16_BLAS_SOLUTIONS
        logger.info("env bootstrap: %s=%s", _PIN_ENV, os.environ[_PIN_ENV])

    # Pinning needs TunableOp enabled and tuning off. Tuning on ignores the
    # pin and retunes every new token count. TunableOp off ignores the pins.
    if os.environ.get(_PIN_ENV, "0") not in ("0", ""):
        os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "1")
        os.environ.setdefault("PYTORCH_TUNABLEOP_TUNING", "0")
        logger.info(
            "env bootstrap: BLAS pins requested -> PYTORCH_TUNABLEOP_ENABLED=%s "
            "PYTORCH_TUNABLEOP_TUNING=%s",
            os.environ["PYTORCH_TUNABLEOP_ENABLED"],
            os.environ["PYTORCH_TUNABLEOP_TUNING"],
        )

    # The FP8 HSTU forward arm, behind ONE switch. Read at import time by
    # ops/triton/triton_fp8.py, so like the flags above it has to reach
    # os.environ before the heavy imports rather than being handed to the model
    # at construction.
    #
    # Off means absent: with the master off NOTHING from _FP8_FORWARD_ARM is
    # exported, so a bf16 run is byte-identical to one where this block does not
    # exist. That is structural rather than incidental -- SAVE_UVQK_LAYERS is
    # parsed on the bf16 path too, by STULayer._allow_saved_fp8_uvqk, which
    # RAISES on a layer no STUStack owns.
    _bind_bool("DLRM_HSTU_FP8_FWD", DLRM_HSTU_FP8_FWD)
    if os.environ.get("DLRM_HSTU_FP8_FWD", "0") == "1":
        # _bind_value honors a pre-set env var over the preset.
        for name, value in _FP8_FORWARD_ARM.items():
            _bind_value(name, value)
        # HSTU_ATTN_FWD=flydsl (the gin default) with this switch on selects
        # the FlyDSL e4m3 attention forward at dispatch time; do not refuse it
        # here.
    else:
        engaged = [
            knob for knob in _FP8_COMPUTE_KNOBS if os.environ.get(knob) == "1"
        ]
        if engaged:
            raise RuntimeError(
                "FP8 HSTU compute knobs are set but the master switch is off: "
                + ", ".join(engaged)
                + ". The arm is a single knob now -- set DLRM_HSTU_FP8_FWD=1 "
                "(gin: apply_env_bootstrap.DLRM_HSTU_FP8_FWD) for the certified "
                "configuration, and override an individual knob only to A/B it "
                "against that arm."
            )

    # The backward is one switch. bwd_fp8.py reads it directly.
    _bind_bool("DLRM_HSTU_FP8_BWD", DLRM_HSTU_FP8_BWD)
    # Attention-backward qk GEMM math. Independent of the projection-dgrad
    # switch above. Default fp8: k128 MFMA on calls whose q and k are already
    # e4m3 (calibration stays bf16 math). Opt out with QK_MFMA=bf16.
    _bind_value("DLRM_HSTU_FP8_BWD_QK_MFMA", DLRM_HSTU_FP8_BWD_QK_MFMA)

    # Each backward unit is read per call by its kernel module (absent = off).
    # C8 always runs with its numerics fix: per-(sequence, head, MX token
    # group) dO scales (DOBLK) and the saturating e4m3 cast (SAT). Those used
    # to be separate knobs; they are now the only C8 path.
    _check_removed_c8_knobs()
    _bind_bool("DLRM_HSTU_FP8_BWD_FULL", DLRM_HSTU_FP8_BWD_FULL)
    _bind_bool("DLRM_HSTU_FP8_MASK16", DLRM_HSTU_FP8_MASK16)
    _bind_value("DLRM_HSTU_FP8_BWD_EPILOGUE", DLRM_HSTU_FP8_BWD_EPILOGUE)
    # C8 dq-split: e4m3 ds workspace + Triton f32 dq kernel instead of bf16
    # dq atomics (ops/flydsl/hstu_dq_split.py), and its intra-pair lookahead.
    _bind_bool("DLRM_HSTU_FP8_BWD_DQ_SPLIT", DLRM_HSTU_FP8_BWD_DQ_SPLIT)
    _bind_value("DLRM_HSTU_FP8_BWD_PIPE", DLRM_HSTU_FP8_BWD_PIPE)
    # Ship-6 items (yambda_5b.gin (13)-(16)); absent = off in code. Not FP8
    # math knobs; the prefix is what launch_slurm.sh forwards.
    # (13) EC cat-fuse + fast position embedding.
    _bind_bool("DLRM_HSTU_FP8_EC_CAT_FUSE", DLRM_HSTU_FP8_EC_CAT_FUSE)
    _bind_bool("DLRM_HSTU_FP8_EC_STRIDED_GRAD", DLRM_HSTU_FP8_EC_STRIDED_GRAD)
    _bind_bool("DLRM_HSTU_FP8_POSENC_FAST", DLRM_HSTU_FP8_POSENC_FAST)
    # (14) Pipelined LayerNorm backward v2 (+ fast sigmoid sub-switch).
    _bind_bool("DLRM_HSTU_FP8_WLN_BWD_V2", DLRM_HSTU_FP8_WLN_BWD_V2)
    _bind_bool(
        "DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID", DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID
    )
    _bind_bool("DLRM_HSTU_FP8_LNMD_BWD_V2", DLRM_HSTU_FP8_LNMD_BWD_V2)
    # (15) Store instead of recompute: normed X and the targets-only VK.
    _bind_bool("DLRM_HSTU_FP8_SAVE_NORMED_X", DLRM_HSTU_FP8_SAVE_NORMED_X)
    _bind_bool("DLRM_HSTU_FP8_SAVE_TARGETS_VK", DLRM_HSTU_FP8_SAVE_TARGETS_VK)
    # (16) Bias-grad column sum (triton_colsum).
    _bind_bool("DLRM_HSTU_FP8_COLSUM", DLRM_HSTU_FP8_COLSUM)
    # Host-sync-free training step (dlrm_v4/host_sync_free.py). Not an FP8
    # knob: the prefix is only what launch_slurm.sh forwards into the
    # container, so DLRM_HSTU_FP8_SYNCFREE=0 in ARM_ENV can switch it off.
    _bind_bool("DLRM_HSTU_FP8_SYNCFREE", DLRM_HSTU_FP8_SYNCFREE)
    # One-time gc.collect()+gc.freeze() before the first window rendezvous
    # (train/utils.py _collect_and_freeze_gc). Not FP8; prefix = forwarding.
    _bind_bool("DLRM_HSTU_FP8_GC_FREEZE", DLRM_HSTU_FP8_GC_FREEZE)
    # FlyDSL attention fwd/bwd launch on torch.cuda.current_stream() instead
    # of HIP's null stream (ops/flydsl/__init__.py current_stream_launch(),
    # used by hstu_fwd.py / hstu_bwd_dkdv.py). Not FP8; prefix = forwarding.
    _bind_bool(
        "DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM", DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM
    )
    # Ship-4 startup items (yambda_5b.gin (9), (10)); absent = off in code.
    # Runtime C++ extension prebuild before RUN_START (train_ranker.py ->
    # ops/triton/triton_fp8.prebuild_extensions). Not FP8 math; prefix =
    # forwarding.
    _bind_bool("DLRM_HSTU_FP8_PREBUILD", DLRM_HSTU_FP8_PREBUILD)
    # Robust Triton autotune pin table (ops/triton/autotune_pin_table.py),
    # applied in train_ranker.py right after the Phase-2 imports.
    _bind_bool("DLRM_HSTU_FP8_AUTOTUNE_PINS", DLRM_HSTU_FP8_AUTOTUNE_PINS)
    # Caching-allocator pre-grow spec before RUN_START (dlrm_v4/alloc_pregrow.py,
    # called from train_ranker.py); "0" = off. Not FP8 math; prefix = forwarding.
    _bind_value("DLRM_HSTU_FP8_ALLOC_PREGROW", DLRM_HSTU_FP8_ALLOC_PREGROW)
    # The sync-free step launches FlyDSL work while torchrec side streams are
    # current; on the legacy null-stream launch that raced its producers (20/20
    # wrong outputs). Refuse the combination at startup.
    if (
        os.environ.get("DLRM_HSTU_FP8_SYNCFREE", "0") == "1"
        # Same deny-list parse as ops/flydsl current_stream_launch(), so the
        # startup check and the launch-time helper can never disagree.
        and os.environ.get("DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM", "1").strip().lower()
        in ("0", "false", "off", "no")
    ):
        raise RuntimeError(
            "DLRM_HSTU_FP8_SYNCFREE=1 requires DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM=1: "
            "the legacy FlyDSL null-stream launch races side-stream producers "
            "(the launch raced its producers and produced wrong outputs). Set "
            "DLRM_HSTU_FP8_SYNCFREE=0 to test the legacy launch stream."
        )

    # An FP8 backward requires the FP8 forward. Opting out of the forward
    # alone is how a non-gfx950 host selects bf16, so both must be off together.
    if (
        os.environ.get("DLRM_HSTU_FP8_BWD", "0") == "1"
        and os.environ.get("DLRM_HSTU_FP8_FWD", "0") != "1"
    ):
        raise RuntimeError(
            "DLRM_HSTU_FP8_BWD=1 with the FP8 forward off. The backward arm is "
            "measured as an increment on the forward arm and that pairing is "
            "the only one with evidence behind it. Set DLRM_HSTU_FP8_BWD=0 as "
            "well to run bf16 (a non-gfx950 host wants both off, alongside "
            "HSTU_ATTN_BWD=triton), or turn the forward back on."
        )
