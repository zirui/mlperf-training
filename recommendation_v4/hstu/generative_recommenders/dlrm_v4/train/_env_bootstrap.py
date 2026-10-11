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

# The FP8 HSTU forward arm: one named knob set, applied as a unit because the
# knobs are only valid in combination (e.g. FUSE_QKV and ATTN together are what
# keep the fused path's memory footprint down).
#
# A pre-set environment variable still wins per knob.
_FP8_FORWARD_ARM: Dict[str, str] = {
    # e4m3 operands for the HSTU linear layers, and for attention q/k/v.
    "DLRM_HSTU_FP8_GEMM": "1",
    "DLRM_HSTU_FP8_ATTN": "1",
    # Fold each quantization into the op that already holds the value in
    # registers rather than quantizing as a separate pass.
    "DLRM_HSTU_FP8_FUSE_LN": "1",
    "DLRM_HSTU_FP8_FUSE_OUTLN": "1",
    "DLRM_HSTU_FP8_FUSE_QKV": "1",
    "DLRM_HSTU_FP8_FUSE_SILU": "1",
    "DLRM_HSTU_FP8_FUSE_RESIDUAL": "1",
    "DLRM_HSTU_FP8_FUSE_RESIDUAL_ALGORITHM": "1",
    # Keep the e4m3 uvqk pack so the backward re-reads it instead of
    # recomputing the projection, on all three HSTU layers. Each pack is
    # large, hence a layer mask.
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
# master switch is a partial arm, so it fails loudly rather than degrading
# silently (e.g. a bf16 run inheriting an FP8 environment from the shell).
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


def _split_tokens(raw: Optional[Union[str, int]]) -> list:
    return [t.strip() for t in str(raw or "").split(",") if t.strip()]


def _add_tokens(
    base: Optional[str], tokens: list
) -> Optional[str]:
    """Append comma tokens to a gin-bound token list.

    An unbound or off list ("", "0", "off", "none") is returned unchanged, so a
    later item never re-arms an earlier item that was switched off.
    """
    if base is None or str(base).strip().lower() in ("", "0", "off", "none"):
        return base
    have = _split_tokens(base)
    return ",".join(have + [t for t in tokens if t not in have])


def _truthy(raw: Optional[str]) -> bool:
    return (raw or "").strip().lower() in ("1", "true", "on", "yes")


# Folded into DLRM_HSTU_FP8_BWD_FULL (C8): block dO scales and the saturating
# cast are its only code path. Kept as names only so a stale opt-out fails
# loudly instead of being silently ignored.
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
    DLRM_HSTU_FP8_BWD_VALU: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_MSPLIT: Optional[int] = None,
    DLRM_HSTU_FP8_BWD_MSPLIT_SCHED: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT: Optional[str] = None,
    DLRM_HSTU_FP8_EC_CAT_FUSE: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_STRIDED_GRAD: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_FAST: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_BWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID: Optional[bool] = None,
    DLRM_HSTU_FP8_LNMD_BWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_SAVE_NORMED_X: Optional[bool] = None,
    DLRM_HSTU_FP8_SAVE_TARGETS_VK: Optional[bool] = None,
    DLRM_HSTU_FP8_COLSUM: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_BWD_FLYDSL: Optional[bool] = None,
    DLRM_HSTU_FP8_FWD_K512: Optional[bool] = None,
    DLRM_HSTU_FP8_FWD_K1536: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WS_LAYOUT: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_WS_AUX: Optional[Union[str, int]] = None,
    DLRM_HSTU_FP8_SYNCFREE: Optional[bool] = None,
    DLRM_HSTU_FP8_GC_FREEZE: Optional[bool] = None,
    DLRM_HSTU_FP8_DENSE_ADAM_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM: Optional[bool] = None,
    DLRM_HSTU_FP8_PREBUILD: Optional[bool] = None,
    DLRM_HSTU_FP8_AUTOTUNE_PINS: Optional[bool] = None,
    DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT: Optional[bool] = None,
    DLRM_HSTU_FP8_ALLOC_PREGROW: Optional[str] = None,
    DLRM_HSTU_FP8_WLN_FWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_LNMD_FWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_CONCAT_FOLD: Optional[bool] = None,
    DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE: Optional[bool] = None,
    DLRM_HSTU_FP8_TGT_SPARSE_RESID: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_GROUP_BWD: Optional[bool] = None,
    DLRM_HSTU_FP8_SEQ_A2A_ROWPERM: Optional[bool] = None,
    DLRM_HSTU_FP8_UNBUCKETIZE_GATHER: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_OUT_WGRAD: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_UVQK: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_UVQK_WGRAD: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_UVQK_STORE_T: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_VK_PRODUCER: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_SORT32: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_GATHER_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_BWD_POW2: Optional[bool] = None,
    DLRM_HSTU_FP8_WSHADOW_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_DDP_BATCHED_COPY: Optional[bool] = None,
    DLRM_HSTU_FP8_TARGET_B: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_STORE_T_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_VKP_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_COLSUM_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_DQ_E5M2: Optional[bool] = None,
    DLRM_HSTU_FP8_SHADOW_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_HELPERS_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_FWD_U_DP: Optional[bool] = None,
    DLRM_HSTU_FP8_FWD_PIPE: Optional[bool] = None,
    DLRM_HSTU_FP8_DELTA_FWD_BM16: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_ATTN_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_D256_V3: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_DXSUM_V3: Optional[bool] = None,
    DLRM_HSTU_FP8_LNMD_FWD_V3: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_REDUCE_LPT: Optional[bool] = None,
    DLRM_HSTU_FP8_LNMD_FWD_MASKGEN: Optional[bool] = None,
    DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_DU_E5M2: Optional[bool] = None,
    DLRM_HSTU_FP8_TM_NONEXP: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WGRAD_CK: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WGRAD_CK_W3: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED: Optional[bool] = None,
    DLRM_HSTU_FP8_TM_NONEXP_TILED: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_DROPOUT_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_SORT16: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_BWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_JOINT_SORT: Optional[bool] = None,
    DLRM_HSTU_FP8_ACTION_ENC_FUSE: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE: Optional[bool] = None,
    DLRM_HSTU_FP8_CONTEXTUAL_FUSE: Optional[bool] = None,
    DLRM_HSTU_FP8_LAZY_FLOPS_STASH: Optional[bool] = None,
    DLRM_HSTU_FP8_IDIST_AFTER_PREV_FWD: Optional[int] = None,
    DLRM_HSTU_FP8_DROPOUT_MASK_BITS: Optional[bool] = None,
    DLRM_HSTU_FP8_CONCAT_FOLD: Optional[bool] = None,
    DLRM_HSTU_FP8_GLUE_FOLD3: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_DROP_HWEPI: Optional[bool] = None,
    DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_POS_ARITH: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_TS_ONESWEEP: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_JOINT_ONESWEEP: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_GATHER_WIDE: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_REDUCE_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_GLUE_FOLD4: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_DKDV_E5M2: Optional[bool] = None,
    DLRM_HSTU_FP8_DOBLK_WL: Optional[bool] = None,
    DLRM_HSTU_FP8_DELTA_BWD_V2_DIET: Optional[bool] = None,
    DLRM_HSTU_FP8_ATTN_GLUE_LAZY: Optional[bool] = None,
    DLRM_HSTU_FP8_FWD_MASK_ARITH: Optional[bool] = None,
    DLRM_HSTU_FP8_FWD_PV: Optional[bool] = None,
    DLRM_HSTU_FP8_PREPROC_L2: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_VF3P: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_FULL_DSC_K: Optional[Union[str, int]] = None,
    TRITON_SPEC_RECORD: Optional[str] = None,
    TRITON_SPEC_REPLAY: Optional[str] = None,
    TRITON_SPEC_REPLAY_PHASES: Optional[str] = None,
    TRITON_SPEC_MONITOR: Optional[Union[str, int, bool]] = None,
) -> None:
    # A pre-set environment variable wins over the gin binding. The pinned
    # triton configs are MI350X-specific, so a different GPU arch (e.g. B200
    # sm_100) sets TRITON_FULL_AUTOTUNE=1 in the launcher environment to
    # re-enable the full autotune search WITHOUT editing this (AMD-default)
    # gin file.
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
    # Triton backward -- see ops/hstu_attention.py.
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
    # BF16 is in the name because the pins are BF16-only: with the FP8 forward
    # arm on, the uvqk_fwd and out_fwd pins name projections that have become
    # torch._scaled_mm and stop applying.
    if _PIN_ENV in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set %s=%s (overrides gin binding)",
            _PIN_ENV, os.environ[_PIN_ENV],
        )
    elif GR_PIN_BF16_BLAS_SOLUTIONS is not None:
        os.environ[_PIN_ENV] = GR_PIN_BF16_BLAS_SOLUTIONS
        logger.info("env bootstrap: %s=%s", _PIN_ENV, os.environ[_PIN_ENV])

    # Pinning only takes effect with TunableOp enabled and its tuning sweep off,
    # so those are set here too rather than left to the caller. With tuning ON,
    # TunableOp sweeps every new token count instead of honouring the pin; with
    # TunableOp OFF the pins are silently ignored.
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
        # _bind_value honors a pre-set env var, so per-knob overrides win
        # over the arm while the rest of it holds.
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
                + ". The arm is a single knob -- set DLRM_HSTU_FP8_FWD=1 "
                "(gin: apply_env_bootstrap.DLRM_HSTU_FP8_FWD) for the default "
                "configuration, and override an individual knob only on top of "
                "that arm."
            )

    # The backward arm is one switch and nothing else. bwd_fp8.py reads it
    # directly and carries the defaults for its tunables, so there is no
    # preset to expand here and no partial arm to fail on.
    _bind_bool("DLRM_HSTU_FP8_BWD", DLRM_HSTU_FP8_BWD)
    # Attention-backward qk GEMM math. Independent of the projection-dgrad
    # switch above. Default fp8: k128 MFMA on calls whose q and k are already
    # e4m3 (calibration stays bf16 math). Opt out with QK_MFMA=bf16.
    _bind_value("DLRM_HSTU_FP8_BWD_QK_MFMA", DLRM_HSTU_FP8_BWD_QK_MFMA)

    # Attention-backward units (C8 full-FP8 backward, mask16, epilogue), each
    # read per call by its kernel module (absent = off).
    # C8 always runs with per-(sequence, head, MX token group) dO scales
    # (DOBLK) and the saturating e4m3 cast (SAT).
    _check_removed_c8_knobs()
    _bind_bool("DLRM_HSTU_FP8_BWD_FULL", DLRM_HSTU_FP8_BWD_FULL)
    _bind_bool("DLRM_HSTU_FP8_MASK16", DLRM_HSTU_FP8_MASK16)
    _bind_value("DLRM_HSTU_FP8_BWD_EPILOGUE", DLRM_HSTU_FP8_BWD_EPILOGUE)
    # C8 dq-split: e4m3 ds workspace + Triton f32 dq kernel instead of bf16
    # dq atomics (ops/flydsl/hstu_dq_split.py), and its intra-pair lookahead.
    _bind_bool("DLRM_HSTU_FP8_BWD_DQ_SPLIT", DLRM_HSTU_FP8_BWD_DQ_SPLIT)
    _bind_value("DLRM_HSTU_FP8_BWD_PIPE", DLRM_HSTU_FP8_BWD_PIPE)
    # Attention-backward VALU diet, m-split body and its schedule, and the
    # dq-split XCD schedule + load options. All read per call by
    # ops/flydsl/hstu_flydsl_mha.py and act on dq-split calls only; absent =
    # off = the base dq-split kernels and compile-cache keys.
    # Attention-backward bitwise bundle on top of those, behind one switch,
    # DLRM_HSTU_FP8_BWD_ATTN_V2: when it is on, the token lists above that are
    # bound by gin gain the V2 tokens and the dq-split ws store layout/aux are
    # bound. A pre-set env var of any of these names still wins verbatim (no
    # tokens are added to it); a list that is off ("0"/empty) stays off.
    # DLRM_HSTU_FP8_BWD_ATTN_V2=0 (or no binding) leaves the lists as bound.
    _bind_bool("DLRM_HSTU_FP8_BWD_ATTN_V2", DLRM_HSTU_FP8_BWD_ATTN_V2)
    _attn_v2 = os.environ.get("DLRM_HSTU_FP8_BWD_ATTN_V2", "0") == "1"
    if _attn_v2:
        _valu_eff = os.environ.get("DLRM_HSTU_FP8_BWD_VALU", DLRM_HSTU_FP8_BWD_VALU)
        _layout_eff = os.environ.get(
            "DLRM_HSTU_FP8_BWD_WS_LAYOUT", DLRM_HSTU_FP8_BWD_WS_LAYOUT or "l128t"
        )
        # nobr needs mc, and mc needs the VALU msat token; dzr needs l128t.
        _msat = "msat" in _split_tokens(_valu_eff)
        _ms_add = ["xt", "rs2", "lc", "ek", "nk", "wh"]
        if _msat:
            _ms_add.append("mc")
        if str(_layout_eff).strip().lower() == "l128t":
            _ms_add.append("dzr")
        _ms_add += ["bst", "ob0", "obj"]
        DLRM_HSTU_FP8_BWD_VALU = _add_tokens(
            DLRM_HSTU_FP8_BWD_VALU, ["nobr"] if _msat else []
        )
        DLRM_HSTU_FP8_BWD_MSPLIT_SCHED = _add_tokens(
            DLRM_HSTU_FP8_BWD_MSPLIT_SCHED, _ms_add
        )
        # kb needs the sp token and a dq-split schedule (dq_from_ds asserts it).
        _sched_eff = os.environ.get(
            "DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED", DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED
        )
        _sched_on = str(_sched_eff or "").strip().lower() not in (
            "", "0", "off", "none")
        if _sched_on and "sp" in _split_tokens(DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT):
            DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT = _add_tokens(
                DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT, ["kb"]
            )
        if DLRM_HSTU_FP8_BWD_WS_LAYOUT is None:
            DLRM_HSTU_FP8_BWD_WS_LAYOUT = "l128t"
        if DLRM_HSTU_FP8_BWD_WS_AUX is None:
            DLRM_HSTU_FP8_BWD_WS_AUX = "2"
    _bind_value("DLRM_HSTU_FP8_BWD_VALU", DLRM_HSTU_FP8_BWD_VALU)
    _bind_value("DLRM_HSTU_FP8_BWD_MSPLIT", DLRM_HSTU_FP8_BWD_MSPLIT)
    _bind_value("DLRM_HSTU_FP8_BWD_MSPLIT_SCHED", DLRM_HSTU_FP8_BWD_MSPLIT_SCHED)
    _bind_value("DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED", DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED)
    _bind_value("DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT", DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT)
    # Absent = off in code. Not FP8 math knobs despite the prefix.
    # EC cat-fuse + fast position embedding.
    _bind_bool("DLRM_HSTU_FP8_EC_CAT_FUSE", DLRM_HSTU_FP8_EC_CAT_FUSE)
    _bind_bool("DLRM_HSTU_FP8_EC_STRIDED_GRAD", DLRM_HSTU_FP8_EC_STRIDED_GRAD)
    _bind_bool("DLRM_HSTU_FP8_POSENC_FAST", DLRM_HSTU_FP8_POSENC_FAST)
    # Pipelined LayerNorm backward v2 (+ fast sigmoid sub-switch).
    _bind_bool("DLRM_HSTU_FP8_WLN_BWD_V2", DLRM_HSTU_FP8_WLN_BWD_V2)
    _bind_bool(
        "DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID", DLRM_HSTU_FP8_WLN_BWD_V2_FAST_SIGMOID
    )
    _bind_bool("DLRM_HSTU_FP8_LNMD_BWD_V2", DLRM_HSTU_FP8_LNMD_BWD_V2)
    # Store instead of recompute: normed X and the targets-only VK.
    _bind_bool("DLRM_HSTU_FP8_SAVE_NORMED_X", DLRM_HSTU_FP8_SAVE_NORMED_X)
    _bind_bool("DLRM_HSTU_FP8_SAVE_TARGETS_VK", DLRM_HSTU_FP8_SAVE_TARGETS_VK)
    # Bias-grad column sum (triton_colsum).
    _bind_bool("DLRM_HSTU_FP8_COLSUM", DLRM_HSTU_FP8_COLSUM)
    # LayerNorm-family folds; absent = off in code. Not FP8 math knobs.
    _bind_bool("DLRM_HSTU_FP8_WLN_FWD_V2", DLRM_HSTU_FP8_WLN_FWD_V2)
    _bind_bool("DLRM_HSTU_FP8_LNMD_FWD_V2", DLRM_HSTU_FP8_LNMD_FWD_V2)
    _bind_bool("DLRM_HSTU_FP8_POSENC_CONCAT_FOLD", DLRM_HSTU_FP8_POSENC_CONCAT_FOLD)
    _bind_bool(
        "DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD", DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD
    )
    _bind_bool("DLRM_HSTU_FP8_TGT_SPARSE_RESID", DLRM_HSTU_FP8_TGT_SPARSE_RESID)
    # The layer-0 input-LayerNorm backward with the dropout mask*scale
    # epilogue on the FlyDSL kernel (ops/flydsl/ln_bwd.py, mscale); needs
    # DLRM_HSTU_FP8_WLN_BWD_FLYDSL as well. Absent = off in code. Not an FP8
    # knob.
    _bind_bool(
        "DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE", DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE
    )
    # FlyDSL weighted LayerNorm backward (ops/flydsl/ln_bwd.py); takes the
    # step's four norm-backward variants ahead of the Triton v1/v2/v3 choice.
    # Not an FP8 knob.
    _bind_bool("DLRM_HSTU_FP8_WLN_BWD_FLYDSL", DLRM_HSTU_FP8_WLN_BWD_FLYDSL)
    # K=512 FP8 forward projection GEMM (ops/hip/k512_fp8_gemm.py): the V/Q/K
    # and targets V/K e4m3-output projections on a HIP kernel instead of
    # hipBLASLt. fp32 summation order only (bitwise except in hipBLASLt's
    # stream-K tail tiles). Read once at triton_fp8 import, so it must be
    # bound here.
    _bind_bool("DLRM_HSTU_FP8_FWD_K512", DLRM_HSTU_FP8_FWD_K512)
    # K=1536 FP8 output-projection forward GEMM + residual
    # (ops/hip/k1536_fp8_addmm.py): the out-projection forward on a HIP kernel
    # instead of the residual-fused hipBLASLt call; fp32 summation order only,
    # as the K=512 GEMM. Read once at triton_fp8 import.
    _bind_bool("DLRM_HSTU_FP8_FWD_K1536", DLRM_HSTU_FP8_FWD_K1536)
    # dq-split ds workspace store layout + store cache-policy aux bits
    # (bitwise; dq-split calls only).
    _bind_value("DLRM_HSTU_FP8_BWD_WS_LAYOUT", DLRM_HSTU_FP8_BWD_WS_LAYOUT)
    _bind_value("DLRM_HSTU_FP8_BWD_WS_AUX", DLRM_HSTU_FP8_BWD_WS_AUX)
    # Host-sync-free training step (dlrm_v4/host_sync_free.py). Not an FP8
    # knob despite the prefix.
    _bind_bool("DLRM_HSTU_FP8_SYNCFREE", DLRM_HSTU_FP8_SYNCFREE)
    # One-time gc.collect()+gc.freeze() before the first window rendezvous
    # (train/utils.py _collect_and_freeze_gc). Not FP8.
    _bind_bool("DLRM_HSTU_FP8_GC_FREEZE", DLRM_HSTU_FP8_GC_FREEZE)
    # The dense Adam step and the clip multiply in one bitwise Triton launch
    # (ops/triton/dense_adam_fused.py). Absent = off in code; read when the
    # dense optimizer is constructed (after this runs). Only "0" disarms.
    # Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_DENSE_ADAM_FUSED", DLRM_HSTU_FP8_DENSE_ADAM_FUSED)
    # FlyDSL attention fwd/bwd launch on torch.cuda.current_stream() instead
    # of HIP's null stream (ops/flydsl/__init__.py current_stream_launch(),
    # used by hstu_fwd.py / hstu_bwd_dkdv.py). Not FP8.
    _bind_bool(
        "DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM", DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM
    )
    # Startup knobs; absent = off in code.
    # Runtime C++ extension prebuild before RUN_START (train_ranker.py ->
    # ops/triton/triton_fp8.prebuild_extensions). Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_PREBUILD", DLRM_HSTU_FP8_PREBUILD)
    # Robust Triton autotune pin table (ops/triton/autotune_pin_table.py),
    # applied in train_ranker.py right after the Phase-2 imports.
    _bind_bool("DLRM_HSTU_FP8_AUTOTUNE_PINS", DLRM_HSTU_FP8_AUTOTUNE_PINS)
    # The posenc concat rows of that table (own switch; absent = off in code,
    # the gin binding turns them on, "0" = kill switch).
    _bind_bool(
        "DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT",
        DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT,
    )
    # Caching-allocator pre-grow spec before RUN_START (dlrm_v4/alloc_pregrow.py,
    # called from train_ranker.py); "0" = off. Not FP8 math.
    _bind_value("DLRM_HSTU_FP8_ALLOC_PREGROW", DLRM_HSTU_FP8_ALLOC_PREGROW)
    # Absent = off in code. Not FP8 math. Two independently switchable parts:
    # grouped EC expand backward (train/ec_bf16_output.py _ECExpandGroup) ...
    _bind_bool("DLRM_HSTU_FP8_EC_GROUP_BWD", DLRM_HSTU_FP8_EC_GROUP_BWD)
    # ... and the sequence-embedding comm permutes (dlrm_v4/seq_a2a_rowperm.py,
    # dlrm_v4/unbucketize_gather.py; installed in MetricsLogger.__init__).
    _bind_bool("DLRM_HSTU_FP8_SEQ_A2A_ROWPERM", DLRM_HSTU_FP8_SEQ_A2A_ROWPERM)
    _bind_bool(
        "DLRM_HSTU_FP8_UNBUCKETIZE_GATHER", DLRM_HSTU_FP8_UNBUCKETIZE_GATHER
    )
    # FP8 output-projection wgrad (ops/triton/triton_hstu_linear.py
    # HSTUComputeOutputFunction; operand cast in ops/triton/bwd_fp8.py). Read
    # at import time; absent = off = the bf16 wgrad. A declared numerics
    # change: the one switch resolves IMPL=nn, DFMT=e4m3, DFMT_SHIFT=auto
    # unless those DLRM_HSTU_FP8_BWD_OUT_WGRAD_* knobs are set explicitly.
    # Needs DLRM_HSTU_FP8_BWD=1 (the e5m2 dout cast it reuses) to engage.
    _bind_bool("DLRM_HSTU_FP8_BWD_OUT_WGRAD", DLRM_HSTU_FP8_BWD_OUT_WGRAD)
    # FP8 uvqk/vk projection backward (ops/triton/bwd_fp8.py); absent = off
    # in code. UVQK is the switch (E5M2 dgrad); WGRAD
    # "fp8t" adds the token-major FP8 wgrad; STORE_T and VK_PRODUCER refine
    # fp8t. Read at bwd_fp8 import time. A numerics change.
    _bind_bool("DLRM_HSTU_FP8_BWD_UVQK", DLRM_HSTU_FP8_BWD_UVQK)
    _bind_value("DLRM_HSTU_FP8_BWD_UVQK_WGRAD", DLRM_HSTU_FP8_BWD_UVQK_WGRAD)
    _bind_bool("DLRM_HSTU_FP8_BWD_UVQK_STORE_T", DLRM_HSTU_FP8_BWD_UVQK_STORE_T)
    _bind_bool("DLRM_HSTU_FP8_BWD_VK_PRODUCER", DLRM_HSTU_FP8_BWD_VK_PRODUCER)
    # Bitwise bundle; absent = off in code, "0" is each part's kill switch.
    # Not FP8 math except WSHADOW_FUSED (same e4m3 bytes). All read at import
    # time except
    # WLN_BWD_POW2 (first LN backward) and DDP_BATCHED_COPY (DMP build).
    # EC backward segment argsort on int32 keys (train/ec_offsets.py) ...
    _bind_bool("DLRM_HSTU_FP8_EC_SORT32", DLRM_HSTU_FP8_EC_SORT32)
    # ... multi-row Triton EC expand gather (train/ec_expand_gather.py) ...
    _bind_bool("DLRM_HSTU_FP8_EC_GATHER_V2", DLRM_HSTU_FP8_EC_GATHER_V2)
    # ... exact 2^-k in the v2 weighted-LN backward (ops/triton/triton_layer_norm.py) ...
    _bind_bool("DLRM_HSTU_FP8_WLN_BWD_POW2", DLRM_HSTU_FP8_WLN_BWD_POW2)
    # ... fused FP8 weight-shadow refresh (ops/triton/triton_fp8_wshadow.py) ...
    _bind_bool("DLRM_HSTU_FP8_WSHADOW_FUSED", DLRM_HSTU_FP8_WSHADOW_FUSED)
    # ... and dense DDP batched_grad_copy (train/utils.py _dense_data_parallel_wrapper).
    _bind_bool("DLRM_HSTU_FP8_DDP_BATCHED_COPY", DLRM_HSTU_FP8_DDP_BATCHED_COPY)
    # FP8 cast/producer diet + U-forward data-parallel GEMM; absent = off in
    # code. Seven independent switches, read at import by bwd_fp8 /
    # triton_layer_norm / hstu_delta_bwd_e5m2 / triton_fp8.
    # DLRM_HSTU_FP8_TARGET_B=0 in the launcher env is the group's kill switch:
    # every switch below that is not itself pre-set is bound to 0 (a pre-set
    # switch still wins).
    _bind_bool("DLRM_HSTU_FP8_TARGET_B", DLRM_HSTU_FP8_TARGET_B)
    _target_b_off = os.environ.get("DLRM_HSTU_FP8_TARGET_B", "1").strip().lower() in (
        "0", "false", "off", "no")
    for _name, _value in (
        ("DLRM_HSTU_FP8_BWD_STORE_T_V2", DLRM_HSTU_FP8_BWD_STORE_T_V2),
        ("DLRM_HSTU_FP8_BWD_VKP_V2", DLRM_HSTU_FP8_BWD_VKP_V2),
        ("DLRM_HSTU_FP8_BWD_COLSUM_V2", DLRM_HSTU_FP8_BWD_COLSUM_V2),
        ("DLRM_HSTU_FP8_BWD_DQ_E5M2", DLRM_HSTU_FP8_BWD_DQ_E5M2),
        ("DLRM_HSTU_FP8_SHADOW_FUSED", DLRM_HSTU_FP8_SHADOW_FUSED),
        ("DLRM_HSTU_FP8_BWD_HELPERS_FUSED", DLRM_HSTU_FP8_BWD_HELPERS_FUSED),
        ("DLRM_HSTU_FP8_FWD_U_DP", DLRM_HSTU_FP8_FWD_U_DP),
    ):
        _bind_bool(_name, False if (_target_b_off and _value is not None) else _value)
    # Two attention kernel configs, each read per call by its kernel module
    # and armed only by "1"; absent = off = the default config.
    # (a) e4m3 FlyDSL attention forward (ops/flydsl/hstu_flydsl_mha.py
    # _FP8_FWD_PIPE_CFG): BM128/waves4 + K/V prefetch + direct P fragment.
    # Bit-identical output for finite V.
    _bind_bool("DLRM_HSTU_FP8_FWD_PIPE", DLRM_HSTU_FP8_FWD_PIPE)
    # (b) Delta-query attention forward (ops/triton/triton_hstu_attention.py
    # triton_cached_hstu_mha, DeltaSize <= 16): BLOCK_M=16 instead of the
    # pinned 128. Bit-identical output.
    _bind_bool("DLRM_HSTU_FP8_DELTA_FWD_BM16", DLRM_HSTU_FP8_DELTA_FWD_BM16)
    # Kernel bundle; absent = off in code. Five independent switches, all
    # read lazily at first use (after this runs).
    _bind_bool("DLRM_HSTU_FP8_D256_V3", DLRM_HSTU_FP8_D256_V3)
    _bind_bool("DLRM_HSTU_FP8_WLN_DXSUM_V3", DLRM_HSTU_FP8_WLN_DXSUM_V3)
    _bind_bool("DLRM_HSTU_FP8_LNMD_FWD_V3", DLRM_HSTU_FP8_LNMD_FWD_V3)
    _bind_bool("DLRM_HSTU_FP8_EC_REDUCE_LPT", DLRM_HSTU_FP8_EC_REDUCE_LPT)
    _bind_bool("DLRM_HSTU_FP8_LNMD_FWD_MASKGEN", DLRM_HSTU_FP8_LNMD_FWD_MASKGEN)
    # Grouped preprocessor LayerNorm(512) tail, forward
    # (ops/preproc_grouped_tail.py). Read when modules/preprocessors.py is
    # imported; absent = off = the per-MLP LayerNorm tails. Bitwise given the
    # Triton autotune pins. Not FP8 math.
    _bind_bool(
        "DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL", DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL
    )
    # FP8 K=tokens weight-gradient bundle; absent = off in code, "0" is each
    # part's kill switch.
    # Four independent switches, two layout switches of (c) / (d) and (g), the
    # placement switch of (b) for those tiled operands (absent = (b)'s pool),
    # all read at import (after this runs) by bwd_fp8 / fp8_tm_alloc /
    # wgrad_ck / wgrad_ck_w3:
    # (a) e5m2 du emitted by the ordinary layer's LN*U*dropout backward
    # (ops/triton/hstu_du_e5m2.py); u-column bias colsum fp32-order.
    _bind_bool("DLRM_HSTU_FP8_BWD_DU_E5M2", DLRM_HSTU_FP8_BWD_DU_E5M2)
    # (b) token-major FP8 wgrad operands from a private plain-segment pool
    # (ops/triton/fp8_tm_alloc.py; prewarmed by dlrm_v4/alloc_pregrow.py).
    # Allocation only, bitwise; inert without expandable_segments:True.
    _bind_bool("DLRM_HSTU_FP8_TM_NONEXP", DLRM_HSTU_FP8_TM_NONEXP)
    # (c) custom uvqk / vk fp8t wgrad (ops/triton/wgrad_ck.py) and (d) custom
    # out-proj NN wgrad (ops/triton/wgrad_ck_w3.py); dW fp32-order,
    # deterministic, first-call guard and latch-off to hipBLASLt.
    _bind_bool("DLRM_HSTU_FP8_BWD_WGRAD_CK", DLRM_HSTU_FP8_BWD_WGRAD_CK)
    _bind_bool("DLRM_HSTU_FP8_BWD_WGRAD_CK_W3", DLRM_HSTU_FP8_BWD_WGRAD_CK_W3)
    # (e) step-tiled xT for (c) (LN store_t / recompute producers,
    # ops/triton/triton_layer_norm.py + wgrad_ck.py) and (f) step-tiled dout_t
    # for (d) (out-proj dual cast, bwd_fp8.py + wgrad_ck_w3.py); operand layout
    # only, bitwise vs the row-major custom path; INERT without (c) / (d).
    _bind_bool(
        "DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED", DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED
    )
    _bind_bool(
        "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED", DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED
    )
    # (g) where (b) places the step-tiled operands of (e) / (f): "1" = (b)'s
    # private pool (also the code default when unset), "0" = the ordinary
    # caching allocator (alloc_pregrow then skips the prewarm when every pooled
    # operand is step-tiled). Allocation only, bitwise; inert without (b) and
    # (e) / (f).
    _bind_bool("DLRM_HSTU_FP8_TM_NONEXP_TILED", DLRM_HSTU_FP8_TM_NONEXP_TILED)
    # The out-proj FP8 dgrad fused with the LN*U*dropout backward
    # (ops/hip/l11_outproj_lnmd.py). Absent = off in code; read lazily at the
    # prebuild / first backward (after this runs). Only "1" arms it;
    # DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED=0 in the environment wins over this
    # binding and restores the unfused chain exactly.
    _bind_bool(
        "DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED", DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED
    )
    # Glue-fusion bundle; absent = off in code, "0" is each part's kill
    # switch. Not FP8 math. All read at import time except
    # IDIST_AFTER_PREV_FWD (train pipeline build).
    # Posenc concat + HSTU input dropout in one kernel (ops/triton/triton_position.py) ...
    _bind_bool(
        "DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE", DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE
    )
    # ... its V2 layout (a refinement: inert while the fusion is off) ...
    _bind_bool("DLRM_HSTU_FP8_POSENC_DROPOUT_V2", DLRM_HSTU_FP8_POSENC_DROPOUT_V2)
    # ... int16 posenc sort keys ...
    _bind_bool("DLRM_HSTU_FP8_POSENC_SORT16", DLRM_HSTU_FP8_POSENC_SORT16)
    # ... posenc backward row-loop kernels (declared fp32-order d_pos/d_ts) ...
    _bind_bool("DLRM_HSTU_FP8_POSENC_BWD_V2", DLRM_HSTU_FP8_POSENC_BWD_V2)
    # ... one argsort for both hoisted EC members (train/ec_offsets.py) ...
    _bind_bool("DLRM_HSTU_FP8_EC_JOINT_SORT", DLRM_HSTU_FP8_EC_JOINT_SORT)
    # ... action encoder, EC merged index and contextual-input glue fusions
    # (ops/glue_fusion.py) ...
    _bind_bool("DLRM_HSTU_FP8_ACTION_ENC_FUSE", DLRM_HSTU_FP8_ACTION_ENC_FUSE)
    _bind_bool(
        "DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE", DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE
    )
    _bind_bool("DLRM_HSTU_FP8_CONTEXTUAL_FUSE", DLRM_HSTU_FP8_CONTEXTUAL_FUSE)
    # ... lazy FLOPs/sample stash (modules/dlrm_hstu.py) ...
    _bind_bool("DLRM_HSTU_FP8_LAZY_FLOPS_STASH", DLRM_HSTU_FP8_LAZY_FLOPS_STASH)
    # ... and the next-batch input-dist placement gate (train/utils.py
    # _idist_gate_resolve): 0 off, 1 single-node runs only, 2 every node count.
    _bind_value(
        "DLRM_HSTU_FP8_IDIST_AFTER_PREV_FWD", DLRM_HSTU_FP8_IDIST_AFTER_PREV_FWD
    )
    # Bitwise glue bundle; absent = off in code, "0" is each part's kill
    # switch. Not FP8 math. All four are read at import time.
    # Input-dropout mask stored as packed bits for the layer-0 LN-backward fold
    # (ops/triton/triton_position.py, ops/triton/triton_layer_norm.py) ...
    _bind_bool("DLRM_HSTU_FP8_DROPOUT_MASK_BITS", DLRM_HSTU_FP8_DROPOUT_MASK_BITS)
    # ... the two int64 timestamp jagged concats as one kernel each
    # (ops/glue_fusion.py section 4; per-site refinements
    # DLRM_HSTU_FP8_CONCAT_FOLD_UIH_TS / _CTX_TS stay default on) ...
    _bind_bool("DLRM_HSTU_FP8_CONCAT_FOLD", DLRM_HSTU_FP8_CONCAT_FOLD)
    # ... the STU parameter casts in one launch plus the lazy attention
    # sort_by_length (ops/glue_fold3.py; per-site refinements
    # DLRM_HSTU_FP8_GLUE_FOLD3_PCAST / _SORT stay default on) ...
    _bind_bool("DLRM_HSTU_FP8_GLUE_FOLD3", DLRM_HSTU_FP8_GLUE_FOLD3)
    # ... and the posenc+dropout V2 kernel's gfx950 hardware bf16 epilogue and
    # v_bitop3 Philox rounds (ops/triton/triton_position.py; a V2 refinement,
    # gfx950 only, self-tested at first use per V2 key). The gin file binds
    # it; the code default stays off.
    _bind_bool("DLRM_HSTU_FP8_POSENC_DROP_HWEPI", DLRM_HSTU_FP8_POSENC_DROP_HWEPI)
    # Grouped preprocessor LayerNorm backward, on the FlyDSL grouped kernel
    # when DLRM_HSTU_FP8_WLN_BWD_FLYDSL owns the three calls (sub-switch
    # DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD_FLYDSL, default on in code;
    # bitwise to the three separate FlyDSL calls). Read lazily at the first
    # backward and by the startup prebuild; absent = off = the per-pair loop;
    # inert while DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL is off. The in-place
    # donation stays opt-in (DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD_INPLACE=1,
    # env only).
    _bind_bool(
        "DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD",
        DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL_BWD,
    )
    # Bitwise sort/gather/glue bundle; absent = off in code, "0" is each
    # part's kill switch. Not FP8 math. All read at import time except
    # EC_REDUCE_FUSED (first use).
    # (a) The posenc pos permutation written from the jagged lengths
    # (ops/triton/triton_pos_perm.py) ...
    _bind_bool("DLRM_HSTU_FP8_POSENC_POS_ARITH", DLRM_HSTU_FP8_POSENC_POS_ARITH)
    # ... (b) the posenc ts-bucket sort and (c) the EC joint sort + offsets on
    # the key-bound onesweep extension (ops/onesweep_sort.py; built before
    # RUN_START under DLRM_HSTU_FP8_PREBUILD) ...
    _bind_bool("DLRM_HSTU_FP8_POSENC_TS_ONESWEEP", DLRM_HSTU_FP8_POSENC_TS_ONESWEEP)
    _bind_bool("DLRM_HSTU_FP8_EC_JOINT_ONESWEEP", DLRM_HSTU_FP8_EC_JOINT_ONESWEEP)
    # ... (d) EC expand gather v2 launch geometry 64 rows x 16 warps (train/
    # ec_expand_gather.py; inert without EC_GATHER_V2) ...
    _bind_bool("DLRM_HSTU_FP8_EC_GATHER_WIDE", DLRM_HSTU_FP8_EC_GATHER_WIDE)
    # ... (e) the grouped EC backward's LPT reduces of both hoisted members in
    # one plan chain + one main launch (train/ec_gpu_plan.py
    # segment_reduce_grouped; inert without EC_REDUCE_LPT) ...
    _bind_bool("DLRM_HSTU_FP8_EC_REDUCE_FUSED", DLRM_HSTU_FP8_EC_REDUCE_FUSED)
    # ... and (f) the small-kernel tail folds (ops/glue_fusion.py section 5;
    # per-site refinements DLRM_HSTU_FP8_GLUE_FOLD4_ACTSUM / _CTX1 / _OFFS /
    # _POST stay default on): bf16 action-encoder masked product + fp32-
    # accumulating sum, contextual pad/cat in one launch per direction,
    # DlrmHSTU.preprocess's integer offsets in one launch, and the
    # targets-only postprocessor time features + cat in one kernel.
    _bind_bool("DLRM_HSTU_FP8_GLUE_FOLD4", DLRM_HSTU_FP8_GLUE_FOLD4)
    # Ordinary-layer bwd_dkdv emits the e5m2 dV|dK columns of the uvqk dz8
    # (ops/flydsl/hstu_dkdv_e5m2.py, called from
    # ops/triton/triton_hstu_preprocess_and_attention.py). Read per call;
    # absent or "0" = off = the separate cast path. A declared fp32-order
    # change of the v|k bias-gradient column sums (payload/amax bitwise).
    _bind_bool("DLRM_HSTU_FP8_BWD_DKDV_E5M2", DLRM_HSTU_FP8_BWD_DKDV_E5M2)
    # Four bitwise attention-adjacent changes, each read per call and armed
    # only by "1"; absent = off.
    # (a) C8 dO block quantizer as a device-built work list (ops/flydsl/
    # hstu_do_fp8_wl.py); e4m3 dO and E8 bytes identical.
    _bind_bool("DLRM_HSTU_FP8_DOBLK_WL", DLRM_HSTU_FP8_DOBLK_WL)
    # (b) VALU-diet form of the VKP_V2 delta-backward kernel
    # (ops/triton/hstu_delta_bwd_e5m2_v2_diet.py), used only in the v2
    # configuration it was verified in; outputs identical to the v2 kernel.
    _bind_bool("DLRM_HSTU_FP8_DELTA_BWD_V2_DIET", DLRM_HSTU_FP8_DELTA_BWD_V2_DIET)
    # (c) sort_by_length indices and the fallback Q/K/V scale tensor built
    # only on the Triton attention fallback path (ops/triton/
    # triton_hstu_preprocess_and_attention.py); removes dead GPU work.
    _bind_bool("DLRM_HSTU_FP8_ATTN_GLUE_LAZY", DLRM_HSTU_FP8_ATTN_GLUE_LAZY)
    # (d) e4m3 FlyDSL attention forward (ops/flydsl/hstu_flydsl_mha.py
    # _FP8_FWD_MASK_ARITH -> ops/flydsl/hstu_fwd.py mask_arith): causal mask
    # of the diagonal n-iterations applied to the packed bf16 P fragment;
    # only together with DLRM_HSTU_FP8_FWD_PIPE. Bit-identical output.
    _bind_bool("DLRM_HSTU_FP8_FWD_MASK_ARITH", DLRM_HSTU_FP8_FWD_MASK_ARITH)
    # FP8 P@V attention forward (ops/flydsl/hstu_flydsl_mha.py _FP8_FWD_PV_CFG,
    # compiled by ops/flydsl/hstu_fwd.py pv="k128" pv_s): e4m3 P with the static
    # P scale s = 10.04 ($DLRM_HSTU_FP8_FWD_PV_SCALE), one f8f6f4 K=128 MFMA per
    # 128 keys; arms only on the FWD_PIPE config (alone or with MASK_ARITH).
    # Declared precision change.
    _bind_bool("DLRM_HSTU_FP8_FWD_PV", DLRM_HSTU_FP8_FWD_PV)
    # FP8 L2 GEMMs of the content/additional preprocessor MLPs plus one stacked
    # LN512 dx-sum backward (ops/preproc_fp8_l2.py). Declared precision
    # change.
    _bind_bool("DLRM_HSTU_FP8_PREPROC_L2", DLRM_HSTU_FP8_PREPROC_L2)
    # vf3p: ops/flydsl/hstu_flydsl_mha.py
    # adds the bwd_dkdv VALU token "vf3p" to the effective
    # DLRM_HSTU_FP8_BWD_VALU set of the dq-split calls (needs ilp; not with
    # vf/vf2/h16/vf3/vf3s; only for 0 < qkv_scale <= 0.25). Read per call;
    # absent or "0" = off = the default act chain. A declared f32
    # reassociation (ulp-level; no format or scale change).
    _bind_bool("DLRM_HSTU_FP8_BWD_VF3P", DLRM_HSTU_FP8_BWD_VF3P)
    # DSC_K (declared numerics change): the C8 attention
    # backward reads DLRM_HSTU_FP8_BWD_FULL_DSC_K (ops/flydsl/hstu_flydsl_mha.py
    # _sc("DSC", "-3"): ds e4m3 cast scale 2**(ceil(log2 N) + K)). A value, not
    # a boolean; env wins; a non-integer fails here instead of at the first
    # backward.
    _bind_value("DLRM_HSTU_FP8_BWD_FULL_DSC_K", DLRM_HSTU_FP8_BWD_FULL_DSC_K)
    if os.environ.get("DLRM_HSTU_FP8_BWD_FULL_DSC_K", "") != "":
        int(os.environ["DLRM_HSTU_FP8_BWD_FULL_DSC_K"])
    # Triton specialization record/replay (dlrm_v4/train/triton_spec.py, read
    # by train_ranker right after this function): record dir, replay table
    # ("0" = off, the gin default), replay phases (post|all) and the
    # post-RUN_START compile monitor. Values, not booleans (_bind_value: a gin
    # bool binds as "True"/"False", which the module also accepts for the
    # monitor). Env wins, as everywhere here.
    _bind_value("TRITON_SPEC_RECORD", TRITON_SPEC_RECORD)
    _bind_value("TRITON_SPEC_REPLAY", TRITON_SPEC_REPLAY)
    _bind_value("TRITON_SPEC_REPLAY_PHASES", TRITON_SPEC_REPLAY_PHASES)
    _bind_value("TRITON_SPEC_MONITOR", TRITON_SPEC_MONITOR)
    # The sync-free step launches FlyDSL work while torchrec side streams are
    # current; the legacy null-stream launch races those producers. Refuse the
    # combination at startup.
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
            "and produces wrong outputs. Set "
            "DLRM_HSTU_FP8_SYNCFREE=0 to test the legacy launch stream."
        )

    # The FP8 backward is only supported on top of the FP8 forward. Since both
    # default ON, opting out of the forward alone (as a non-gfx950 host does)
    # would land on an unsupported bf16-forward/FP8-backward combination.
    # Raise rather than run it: the same fail-closed rule the rest of this arm
    # follows.
    if (
        os.environ.get("DLRM_HSTU_FP8_BWD", "0") == "1"
        and os.environ.get("DLRM_HSTU_FP8_FWD", "0") != "1"
    ):
        raise RuntimeError(
            "DLRM_HSTU_FP8_BWD=1 with the FP8 forward off. The backward arm is "
            "supported only as an increment on the forward arm. "
            "Set DLRM_HSTU_FP8_BWD=0 as "
            "well to run bf16 (a non-gfx950 host wants both off, alongside "
            "HSTU_ATTN_BWD=triton), or turn the forward back on."
        )
