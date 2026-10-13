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
import sys
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
    # use_fast_accum=False. On ROCm (torch 2.12) tensor-wise torch._scaled_mm
    # ignores the flag and hipBLASLt accumulates in fp32 either way; =1 only
    # makes K512 decline.
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


def _group_spans_nodes() -> Optional[bool]:
    """World size vs the ranks of one node, if a process group already exists.

    Looked up in sys.modules, so this module still imports no torch; in
    train_ranker the bootstrap runs before the group exists (None)."""
    dist = sys.modules.get("torch.distributed")
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            local = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0)
            if local <= 0:
                torch = sys.modules.get("torch")
                local = max(1, torch.cuda.device_count()) if torch is not None else 1
            return dist.get_world_size() > local
    except Exception:  # noqa: BLE001
        pass
    return None


def _spans_nodes() -> bool:
    """True when the job runs on more than one node: the multi-node rule of
    _bind_bool_multinode (its default, decided here) and of
    host_sync_free._spans_nodes (which calls it), so keep it torch-free and
    free of package imports. host_sync_free._multi_node (log only)
    and train/utils._idist_gate_topology (idist gate) keep their own.

    NNODES (exported by the launcher at every node count, before the
    trainer starts) or SLURM_NNODES decides when set; only without either is
    an existing process group compared with the ranks of one node. Neither
    and no group (e.g. a bare unit test): False, the single-node answer.
    """
    for key in ("NNODES", "SLURM_NNODES"):
        v = os.environ.get(key, "").strip()
        if v:
            try:
                return int(v) > 1
            except ValueError:
                pass
    return bool(_group_spans_nodes())


def _bind_bool_multinode(name: str, value: Optional[bool]) -> None:
    """`_bind_bool` for a switch whose gin True means ON ONLY WHEN THE JOB SPANS
    MORE THAN ONE NODE.

    True binds "1" when _spans_nodes(), else "0"; False binds "0"; None binds
    nothing. A pre-set env var wins at any node count: "1" forces the part on
    (also at one node), "0" is its kill switch.
    """
    if name in os.environ:
        logger.info(
            "env bootstrap: honoring pre-set %s=%s (overrides gin binding)",
            name,
            os.environ[name],
        )
    elif value is not None:
        on = bool(value) and _spans_nodes()
        os.environ[name] = "1" if on else "0"
        logger.info(
            "env bootstrap: %s=%s%s",
            name,
            os.environ[name],
            ""
            if not value
            else (
                " (multi-node default)"
                if on
                else " (multi-node default: single node, off)"
            ),
        )


def _split_tokens(raw: Optional[Union[str, int]]) -> list:
    return [t.strip() for t in str(raw or "").split(",") if t.strip()]


def _add_tokens(
    base: Optional[str], tokens: list
) -> Optional[str]:
    """Append comma tokens to a gin-bound token list.

    An unbound or off list ("", "0", "off", "none") is returned unchanged, so
    adding tokens never re-arms a list that was switched off.
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
    DLRM_HSTU_FP8_ARM_PRINT_ATOMIC: Optional[bool] = None,
    GR_PIN_BF16_BLAS_SOLUTIONS: Optional[str] = None,
    DLRM_HSTU_FP8_FWD: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_QK_MFMA: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_FULL: Optional[bool] = None,
    DLRM_HSTU_FP8_MASK16: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_EPILOGUE: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_VF4: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_DQ_SPLIT: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_PIPE: Optional[int] = None,
    DLRM_HSTU_FP8_BWD_VALU: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_MSPLIT: Optional[int] = None,
    DLRM_HSTU_FP8_BWD_MSPLIT_SCHED: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_DQ_SPLIT_SCHED: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_DQ_SPLIT_OPT: Optional[str] = None,
    DLRM_HSTU_FP8_DQ8_DIET: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_DO8: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_DO8_TAKE: Optional[bool] = None,
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
    DLRM_HSTU_FP8_FWD_MXFP6: Optional[bool] = None,
    DLRM_HSTU_FP8_BIAS_CACHE_STRICT: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WS_LAYOUT: Optional[str] = None,
    DLRM_HSTU_FP8_BWD_WS_AUX: Optional[Union[str, int]] = None,
    DLRM_HSTU_FP8_SYNCFREE: Optional[bool] = None,
    DLRM_HSTU_FP8_GC_FREEZE: Optional[bool] = None,
    DLRM_HSTU_FP8_PROF_POST_GC: Optional[bool] = None,
    DLRM_HSTU_FP8_WARMUP_F2E: Optional[bool] = None,
    DLRM_HSTU_FP8_WARMUP_FIN_RPP: Optional[bool] = None,
    DLRM_HSTU_FP8_DENSE_ADAM_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM: Optional[bool] = None,
    DLRM_HSTU_FP8_PREBUILD: Optional[bool] = None,
    DLRM_HSTU_FP8_AUTOTUNE_PINS: Optional[bool] = None,
    DLRM_HSTU_FP8_AUTOTUNE_PIN_POSENC_CONCAT: Optional[bool] = None,
    DLRM_HSTU_FP8_ALLOC_PREGROW: Optional[str] = None,
    DLRM_HSTU_FP8_HIP_PRELOAD: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_FWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_LNMD_FWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_CONCAT_FOLD: Optional[bool] = None,
    DLRM_HSTU_FP8_EVAL_FWD_HYGIENE: Optional[bool] = None,
    DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS: Optional[bool] = None,
    DLRM_HSTU_FP8_DROPOUT_MSCALE_FOLD: Optional[bool] = None,
    DLRM_HSTU_FP8_WLN_BWD_FLYDSL_MSCALE: Optional[bool] = None,
    DLRM_HSTU_FP8_TGT_SPARSE_RESID: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_GROUP_BWD: Optional[bool] = None,
    DLRM_HSTU_FP8_SEQ_A2A_ROWPERM: Optional[bool] = None,
    DLRM_HSTU_FP8_SEQ_A2A_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND: Optional[bool] = None,
    DLRM_HSTU_FP8_SEQ_A2A_FUSED_FWD: Optional[bool] = None,
    DLRM_HSTU_FP8_SEQ_A2A_FUSED_BWD: Optional[bool] = None,
    DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV: Optional[bool] = None,
    DLRM_HSTU_FP8_UNBUCKETIZE_GATHER: Optional[bool] = None,
    DLRM_HSTU_FP8_KJT_RECAT_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_TBE_REBAG: Optional[bool] = None,
    DLRM_HSTU_FP8_TBE_BOUNDS_TRITON: Optional[bool] = None,
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
    DL_COLD_PREFETCH_STREAM: Optional[str] = None,
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
    DLRM_HSTU_FP8_EVAL_LNMD_V3: Optional[bool] = None,
    DLRM_HSTU_FP8_PREPROC_GROUPED_TAIL: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_DU_E5M2: Optional[bool] = None,
    DLRM_HSTU_FP8_TM_NONEXP: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WGRAD_CK: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WGRAD_CK_W3: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WGRAD_CK_XT_TILED: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_TILED: Optional[bool] = None,
    DLRM_HSTU_FP8_TM_NONEXP_TILED: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED: Optional[bool] = None,
    DLRM_HSTU_FP8_L11_DO8: Optional[bool] = None,
    DLRM_HSTU_FP8_L11_DO8_ONLY: Optional[bool] = None,
    DLRM_HSTU_FP8_L11_DO8_DIET: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_DROPOUT_FUSE: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_DROPOUT_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_SORT16: Optional[bool] = None,
    DLRM_HSTU_FP8_POSENC_BWD_V2: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_JOINT_SORT: Optional[bool] = None,
    DLRM_HSTU_FP8_ACTION_ENC_FUSE: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_MERGED_INDEX_FUSE: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_INDEXED_L1: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_INDEXED_L1_FP8: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_INDEXED_L1_FP8_BWD: Optional[str] = None,
    DLRM_HSTU_FP8_EC_IX_WGRAD_IDX_AFTER_DOT: Optional[bool] = None,
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
    DLRM_HSTU_FP8_ATTN_INTGLUE: Optional[bool] = None,
    DLRM_HSTU_FP8_DOBLK_WL: Optional[bool] = None,
    DLRM_HSTU_FP8_DELTA_BWD_V2_DIET: Optional[bool] = None,
    DLRM_HSTU_FP8_ATTN_GLUE_LAZY: Optional[bool] = None,
    DLRM_HSTU_FP8_FWD_MASK_ARITH: Optional[bool] = None,
    DLRM_HSTU_FP8_MASKGEN_PK: Optional[bool] = None,
    DLRM_HSTU_FP8_MASKGEN_PK_LAYERS: Optional[str] = None,
    DLRM_HSTU_FP8_L11_VD: Optional[bool] = None,
    DLRM_HSTU_FP8_FWD_PV: Optional[bool] = None,
    DLRM_HSTU_FP8_PREPROC_L2: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE: Optional[bool] = None,
    DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_VF3P: Optional[bool] = None,
    DLRM_HSTU_FP8_BWD_FULL_DSC_K: Optional[Union[str, int]] = None,
    DLRM_HSTU_FP8_FWD_K512_TILED_A: Optional[bool] = None,
    TRITON_SPEC_RECORD: Optional[str] = None,
    TRITON_SPEC_REPLAY: Optional[str] = None,
    TRITON_SPEC_REPLAY_PHASES: Optional[str] = None,
    TRITON_SPEC_MONITOR: Optional[Union[str, int, bool]] = None,
    DLRM_HSTU_FP8_ACTION_PREFILL_L2: Optional[bool] = None,
    DLRM_HSTU_FP8_ACTION_PREFILL: Optional[bool] = None,
    DLRM_HSTU_FP8_DD_OWN_H2D: Optional[bool] = None,
    DLRM_HSTU_FP8_EVAL_EMB_LOOKAHEAD: Optional[bool] = None,
    DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN: Optional[int] = None,
    DLRM_HSTU_FP8_STAGE_REF_ONLY: Optional[bool] = None,
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
    # ops/utils.arm_print_all_ranks writes each arming line as ONE
    # os.write of "\n<msg> [rank N]\n" so the launcher's hipBLASLt chatter
    # filter cannot take it out with a torn warning token; "0" = the
    # plain print(). Logging only. Read per call.
    _bind_bool("DLRM_HSTU_FP8_ARM_PRINT_ATOMIC", DLRM_HSTU_FP8_ARM_PRINT_ATOMIC)

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
    # Streaming cold-page-cache read-ahead (dlrm_v4/train/
    # cold_prefetch.py): "1" (every node count), "auto" (only when $NNODES > 1)
    # or "0". A value, read by cold_prefetch.start() after RUN_START; no gin
    # binding = the code default (off). Env wins, as everywhere here.
    _bind_value("DL_COLD_PREFETCH_STREAM", DL_COLD_PREFETCH_STREAM)

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
    # dq8 epilogue diet (ops/triton/hstu_dq_e5m2.py): the dq-split
    # _dq8_sched_kernel's full-tile epilogue without the row mask (byte-identical
    # outputs). Code default off; the gin file binds it on; only "1" arms it.
    # Read once at hstu_dq_e5m2 import.
    _bind_bool("DLRM_HSTU_FP8_DQ8_DIET", DLRM_HSTU_FP8_DQ8_DIET)
    # dO8 consumer hook (ops/flydsl/hstu_do8_registry.py).
    # "1": the ordinary-layer attention forward creates a dO8 site on
    # training calls whose FlyDSL backward plans the C8 do_blk dO
    # quantization, and that backward takes the e4m3 dO + E8 scales a
    # d(attn) producer published for the dO it received instead of running
    # quantize_do_e4m3_blk (the same bytes by the producer's contract; any
    # failed check quantizes as before). Read per call. Only "1" arms (a
    # True binding writes "1"); absent or "0" = off = no site, no take,
    # no registry line; any other value = off with one line. Inert without
    # a producer: bind it together with the producer's switch
    # (DLRM_HSTU_FP8_L11_DO8). Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_BWD_DO8", DLRM_HSTU_FP8_BWD_DO8)
    # The same hook's retired v1 switch. It can no longer turn anything on: "" or "1" behaves as
    # unset. While DLRM_HSTU_FP8_BWD_DO8 is "1", any other value turns the
    # consumer off with the effect of DLRM_HSTU_FP8_BWD_DO8=0 (no sites, no
    # takes) plus one "retired v1 kill switch" line per process. The binding
    # is True = "1"; a pre-set environment value wins.
    _bind_bool("DLRM_HSTU_FP8_BWD_DO8_TAKE", DLRM_HSTU_FP8_BWD_DO8_TAKE)
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
    # Eval-forward hygiene (ops/eval_fwd_hygiene.py): no-grad (eval) forwards
    # take the posenc folds above and skip work only a backward or the training
    # telemetry reads. Read per call; absent = off. Bitwise outputs. The _LOSS
    # sub-switch (acts only with the main one on) also skips the eval-time
    # multitask loss, which nothing reads in eval; the gin binds it off. Not
    # FP8 math.
    _bind_bool("DLRM_HSTU_FP8_EVAL_FWD_HYGIENE", DLRM_HSTU_FP8_EVAL_FWD_HYGIENE)
    _bind_bool(
        "DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS", DLRM_HSTU_FP8_EVAL_FWD_HYGIENE_LOSS
    )
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
    # MXFP6 forward K=512 trio (ops/hip/mx6_fwd.py): the training forward's
    # LN store_t + U + V/Q/K and the targets-only LN store_t + V/K on the MX6
    # LN producer and the MXFP6 GEMMs. A DECLARED PRECISION CHANGE, bound by
    # the gin file (absent = off in code); only "1" arms it. Read once at
    # import by triton_hstu_preprocess_and_attention / triton_fp8 / mx6_fwd.
    _bind_bool("DLRM_HSTU_FP8_FWD_MXFP6", DLRM_HSTU_FP8_FWD_MXFP6)
    # Scaled-bias cache of the FP8 forward (ops/triton/triton_fp8.py
    # _scale_bias_for_fp8_output): absent or "1" = a hit is served only while a
    # weak reference shows the very bias tensor it came from is alive and is
    # the caller's, dead or updated entries purged at a fixed bound; "0" = the
    # unchecked cache. Bit-identical except on a call where the
    # unchecked cache would have returned a stale entry of another, freed
    # tensor (recomputed there: a declared correction). Read lazily at its
    # first call (after this runs).
    _bind_bool("DLRM_HSTU_FP8_BIAS_CACHE_STRICT", DLRM_HSTU_FP8_BIAS_CACHE_STRICT)
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
    # Profiler post-export full GC (dlrm_v4/utils.py Profiler.step; traced runs
    # only, the Profiler exists only with OUTPUT_TRACE=1): one gc.collect() on
    # every rank right after the export of a trace window that no later window
    # follows. Absent or "1" = on (code default); "0" = kill switch. Read at
    # each export. Not FP8.
    _bind_bool("DLRM_HSTU_FP8_PROF_POST_GC", DLRM_HSTU_FP8_PROF_POST_GC)
    # FP8 first-use warm-up item F2e (dlrm_v4/train/fp8_warmup.py): the e8
    # bwd_dkdv build of DLRM_HSTU_FP8_BWD_DKDV_E5M2 compiled before RUN_START
    # on synthetic inputs. Absent or "1" = run it whenever the warm-up runs
    # with DLRM_HSTU_FP8_BWD_DKDV_E5M2 on (code default); "0" = kill switch
    # (as FP8_WARMUP_SKIP=F2e). Read at warm-up time. Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_WARMUP_F2E", DLRM_HSTU_FP8_WARMUP_F2E)
    # FP8 first-use warm-up item F2r (dlrm_v4/train/fp8_warmup.py): the two
    # e5m2 sink finalizers compiled before RUN_START at every RPP bucket, on
    # synthetic inputs. Absent or "1" = run it whenever the warm-up runs with a
    # sink on (code default); "0" = kill switch (as FP8_WARMUP_SKIP=F2r). Read
    # at warm-up time. Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_WARMUP_FIN_RPP", DLRM_HSTU_FP8_WARMUP_FIN_RPP)
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
    # Targeted HIP code-object preload (dlrm_v4/train/hip_preload.py, read
    # once by the train_ranker hook right after the pre-grow above, after this
    # runs). Absent = off in code;
    # the gin binding True writes "1" (the checked-in table, this run shape's
    # rows); "0" in the environment is the kill switch. A pre-set value wins
    # verbatim, so the env-only values "all" and a table path pass through.
    # Bitwise: no kernel launch, no tensor. Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_HIP_PRELOAD", DLRM_HSTU_FP8_HIP_PRELOAD)
    # Absent = off in code. Not FP8 math. Two independently switchable parts:
    # grouped EC expand backward (train/ec_bf16_output.py _ECExpandGroup) ...
    _bind_bool("DLRM_HSTU_FP8_EC_GROUP_BWD", DLRM_HSTU_FP8_EC_GROUP_BWD)
    # ... and the sequence-embedding comm permutes (dlrm_v4/seq_a2a_rowperm.py,
    # dlrm_v4/unbucketize_gather.py; installed in MetricsLogger.__init__).
    _bind_bool("DLRM_HSTU_FP8_SEQ_A2A_ROWPERM", DLRM_HSTU_FP8_SEQ_A2A_ROWPERM)
    # Four sparse-path parts: part (a), the
    # fused seq-a2a path, binds through _bind_bool, so the gin True turns it
    # on at EVERY node count, one node included (there the table-/column-wise
    # sequence all-to-alls run it; their table-wise owners arrive last at the
    # forward all-to-all). Parts (b)-(d) below bind through
    # _bind_bool_multinode: on only when the job spans more than one node,
    # "0" at one node. A pre-set env var wins for all four ("1" forces a part
    # on, "0" is its kill switch). Absent = off in code.
    # Fused send/receive path of the same permutes (dlrm_v4/seq_a2a_fused.py,
    # used by seq_a2a_rowperm.py; needs the ROWPERM switch).
    _bind_bool("DLRM_HSTU_FP8_SEQ_A2A_FUSED", DLRM_HSTU_FP8_SEQ_A2A_FUSED)
    # Its sub-switches, read once at install (seq_a2a_rowperm.fused_config /
    # recv_config): ..._FUSED_SEND (K1 + K2; ..._FUSED_FWD = K1, ..._FUSED_BWD
    # = K2) and ..._FUSED_RECV (K3' + K4: decode + unbucketize and its
    # backward, patched All2All_Seq_Req_Wait). Absent = on in code, and the
    # gin binds True: the same "1". Plain _bind_bool: the code ANDs each with
    # the switch above, so a sub-switch never turns a part on by itself and
    # the node-count default stays with that switch alone. A pre-set "0" is a
    # part's kill switch. Diagnostic ..._FUSED_VERIFY=N stays env only (gin
    # "diagnostic env knobs (not set in production)" block).
    _bind_bool("DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND", DLRM_HSTU_FP8_SEQ_A2A_FUSED_SEND)
    _bind_bool("DLRM_HSTU_FP8_SEQ_A2A_FUSED_FWD", DLRM_HSTU_FP8_SEQ_A2A_FUSED_FWD)
    _bind_bool("DLRM_HSTU_FP8_SEQ_A2A_FUSED_BWD", DLRM_HSTU_FP8_SEQ_A2A_FUSED_BWD)
    _bind_bool("DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV", DLRM_HSTU_FP8_SEQ_A2A_FUSED_RECV)
    _bind_bool(
        "DLRM_HSTU_FP8_UNBUCKETIZE_GATHER", DLRM_HSTU_FP8_UNBUCKETIZE_GATHER
    )
    # b1pp sparse-chain removals (dlrm_v4/b1pp.py), three independent
    # switches: the fused KJT recat (+ KJT offsets) in host_sync_free's
    # dist_init, the Rkeep rebag and the Triton bounds check of the
    # sequence-embedding TBE (patched BatchedFusedEmbedding.forward, installed
    # in MetricsLogger.__init__). Read once at first use; env-only diagnostic
    # DLRM_HSTU_FP8_B1PP_VERIFY=N.
    _bind_bool_multinode(
        "DLRM_HSTU_FP8_KJT_RECAT_FUSED", DLRM_HSTU_FP8_KJT_RECAT_FUSED
    )
    _bind_bool_multinode("DLRM_HSTU_FP8_TBE_REBAG", DLRM_HSTU_FP8_TBE_REBAG)
    _bind_bool_multinode(
        "DLRM_HSTU_FP8_TBE_BOUNDS_TRITON", DLRM_HSTU_FP8_TBE_BOUNDS_TRITON
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
    # Eval LN*U*dropout forward on the LNMD_FWD_V3 kernel with TRAINING=False
    # (ops/eval_lnmd_v3.py): training=False calls with the v3 eligibility
    # leave the per-row kernel. DECLARED numerics change of the eval pass (not
    # bitwise to the per-row kernel); the gin binds it off. Read per call,
    # only on training=False calls; absent = off.
    _bind_bool("DLRM_HSTU_FP8_EVAL_LNMD_V3", DLRM_HSTU_FP8_EVAL_LNMD_V3)
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
    # and (h), the kernel-tile switch of (f),
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
    # (h) the step-tiled dout_t of (f) read by (d)'s retiled entry
    # wgrad_ck_w3_t256s (ops/triton/wgrad_ck_w3.py) instead of wgrad_ck_w3_t:
    # wider output tiles over more token splits; dW fp32-order vs (f), not
    # bitwise; deterministic, own first-call guard; INERT without (d) and (f).
    _bind_bool(
        "DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S", DLRM_HSTU_FP8_BWD_WGRAD_CK_W3_T256S
    )
    # The out-proj FP8 dgrad fused with the LN*U*dropout backward
    # (ops/hip/l11_outproj_lnmd.py). Absent = off in code; read lazily at the
    # prebuild / first backward (after this runs). Only "1" arms it;
    # DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED=0 in the environment wins over this
    # binding and restores the unfused chain exactly.
    _bind_bool(
        "DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED", DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED
    )
    # The fused out-proj kernel's dO8/E8 emission for the C8 attention
    # backward (ops/hip/l11_do8.py). Absent = off in code; read lazily at the
    # prebuild / first forward (after this runs). Only "1" arms it;
    # DLRM_HSTU_FP8_L11_DO8=0 in the environment wins over this binding and
    # restores the path without it exactly. Inert without
    # DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED, DLRM_HSTU_FP8_BWD_DU_E5M2 and the
    # consumer knob DLRM_HSTU_FP8_BWD_DO8.
    _bind_bool("DLRM_HSTU_FP8_L11_DO8", DLRM_HSTU_FP8_L11_DO8)
    # DO8 'only' mode (ops/hip/l11_do8.py with ops/flydsl/hstu_do8_registry.py):
    # a DO8 call the consumer's plan() answers "only" for skips the bf16
    # d(attn) store. Absent = off in code; read per call (forward and
    # backward), so it binds before the first forward. Only "1" arms it;
    # DLRM_HSTU_FP8_L11_DO8_ONLY=0 in the environment wins over this binding
    # and restores DO8 'plus' exactly. Inert without DLRM_HSTU_FP8_L11_DO8 and
    # its consumer knob.
    _bind_bool("DLRM_HSTU_FP8_L11_DO8_ONLY", DLRM_HSTU_FP8_L11_DO8_ONLY)
    # DO8 diet: the VALU/SGPR-trimmed form of DO8's emission tail
    # (ops/hip/l11_do8.cu, preprocessor switch L11DO8_DIET; ops/hip/l11_do8.py).
    # Absent = off in code (the build adds -DL11DO8_DIET=0); read once at the
    # first DO8 build (the prebuild, after this runs). Only "1" arms it;
    # DLRM_HSTU_FP8_L11_DO8_DIET=0 in the environment wins over this binding.
    # Same output bytes either way. Inert without DLRM_HSTU_FP8_L11_DO8.
    _bind_bool("DLRM_HSTU_FP8_L11_DO8_DIET", DLRM_HSTU_FP8_L11_DO8_DIET)
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
    # intglue (ops/flydsl/hstu_flydsl_mha.py _attn_intglue_reason ->
    # ops/flydsl/hstu_dq_split.py build_workspace_intglue): the dq-split
    # attention backward's int32 seq offsets and ds-workspace tile-prefix
    # header in one Triton launch per call instead of several torch kernels.
    # Read per call; absent or "0" = the torch ops. Bitwise (index bookkeeping).
    _bind_bool("DLRM_HSTU_FP8_ATTN_INTGLUE", DLRM_HSTU_FP8_ATTN_INTGLUE)
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
    # MASKGEN_PK (code default off): the LNMD_FWD_MASKGEN mask-generating
    # LN*U*dropout forward on its bitwise VALU-diet kernel
    # (ops/triton/triton_hstu_linear.py _ln_mul_dropout_fwd_maskgen_pk, gate
    # _use_lnmd_fwd_maskgen_pk). Read lazily at the first armed maskgen call;
    # absent or "0" = off = the base maskgen kernel; inert while
    # DLRM_HSTU_FP8_LNMD_FWD_MASKGEN is off.
    _bind_bool("DLRM_HSTU_FP8_MASKGEN_PK", DLRM_HSTU_FP8_MASKGEN_PK)
    # MASKGEN_PK layer list: comma-separated zero-based STU layer indices
    # (modules/stu.py tags each STULayer's compute output with its STUStack
    # index) on which MASKGEN_PK may run; unset or empty = every layer. A
    # value, not a boolean; env wins; a token that is not a non-negative
    # integer fails here, at startup, instead of at the first forward.
    _bind_value("DLRM_HSTU_FP8_MASKGEN_PK_LAYERS", DLRM_HSTU_FP8_MASKGEN_PK_LAYERS)
    for _tok in _split_tokens(os.environ.get("DLRM_HSTU_FP8_MASKGEN_PK_LAYERS")):
        if not (_tok.isascii() and _tok.isdigit()):
            raise ValueError(
                "DLRM_HSTU_FP8_MASKGEN_PK_LAYERS must be comma-separated "
                f"zero-based STU layer indices, got {_tok!r}"
            )
    # L11_VD (code default off): the BWD_OUTPROJ_LNMD_FUSED kernel built
    # from its VALU-diet source ops/hip/l11_outproj_lnmd_vd.cu
    # (l11_outproj_lnmd.vd_requested). Read at that kernel's build; only "1"
    # selects it; absent, "0" or any other value = ops/hip/l11_outproj_lnmd.cu.
    # Inert while DLRM_HSTU_FP8_BWD_OUTPROJ_LNMD_FUSED is off.
    _bind_bool("DLRM_HSTU_FP8_L11_VD", DLRM_HSTU_FP8_L11_VD)
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
    # The preprocessor L1 dgrads fused into the grouped EC
    # backward (ops/triton/triton_ec_dgrad_segreduce.py, dlrm_v4/train/
    # ec_dgrad_ports.py). Read at first use; absent = off in code. DECLARED
    # fp32 reduction-order numerics change; "0" in the environment is the kill switch.
    _bind_bool("DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE", DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE)
    # X2H: the fused EC dgrad's two passes on the persistent HIP kernel x2h_pass
    # (ops/hip/ec_dgrad_x2h.py + x2h_pass.cu), bit-identical to the Triton pass, armed per device
    # by a pre-RUN_START bitwise cross-check. Read at first use; absent = off in code; acts only
    # while DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE is armed. No numerics change; "0" in the environment
    # is the kill switch.
    _bind_bool("DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP", DLRM_HSTU_FP8_EC_DGRAD_SEGREDUCE_HIP)
    # Indexed-X L1: the content/additional L1 forward and wgrad read X through the EC
    # index from the bf16 unique-row table (ops/ec_indexed_l1.py, ops/triton/triton_ec_indexed_l1.py). Read at
    # first use; absent = off in code. DECLARED fp32 reduction-order numerics change; "0" in the environment is
    # the kill switch.
    _bind_bool("DLRM_HSTU_FP8_EC_INDEXED_L1", DLRM_HSTU_FP8_EC_INDEXED_L1)
    # The Indexed-X L1 TRAINING forward on FP8 e4m3 operands (U8 per-row pow2 x W1_8
    # per-tensor amax/448; ops/triton/triton_ec_indexed_l1_fp8.py, hooks in ops/ec_indexed_l1.py). Read at first
    # use; absent = off in code; acts only while Indexed-X is armed. DECLARED precision change (fp8 e4m3, train
    # only; eval stays bf16); "0" in the environment is the kill switch.
    _bind_bool("DLRM_HSTU_FP8_EC_INDEXED_L1_FP8", DLRM_HSTU_FP8_EC_INDEXED_L1_FP8)
    # Its backward form: "consistent" (absent = this: wgrad / dgrad on the dequantized operands Ud / Wd, the
    # gradient of the quantized forward) or "ste" (diagnostic: the bf16 U / W1). A value, not a boolean; env
    # wins; while the knob above is on, any other value fails here, at startup (the module also refuses it at
    # first use).
    _bind_value("DLRM_HSTU_FP8_EC_INDEXED_L1_FP8_BWD", DLRM_HSTU_FP8_EC_INDEXED_L1_FP8_BWD)
    if _truthy(os.environ.get("DLRM_HSTU_FP8_EC_INDEXED_L1_FP8")) and os.environ.get(
        "DLRM_HSTU_FP8_EC_INDEXED_L1_FP8_BWD", ""
    ).strip().lower() not in ("", "consistent", "ste"):
        raise ValueError(
            "DLRM_HSTU_FP8_EC_INDEXED_L1_FP8_BWD must be 'consistent' or 'ste', got "
            f"{os.environ['DLRM_HSTU_FP8_EC_INDEXED_L1_FP8_BWD']!r}"
        )
    # IX wgrad index order: Indexed-X's split-K wgrad loads the next block's indices after the current
    # block's dot (_ecix_wgrad) instead of one iteration ahead (_ecix_wgrad_idxpf) in
    # ops/triton/triton_ec_indexed_l1.py. Read once, at the first wgrad / prebuild; absent = off in code
    # (_ecix_wgrad_idxpf); acts only while Indexed-X is armed. Bitwise identical P / dW1; "0" in the environment
    # is the kill switch.
    _bind_bool("DLRM_HSTU_FP8_EC_IX_WGRAD_IDX_AFTER_DOT", DLRM_HSTU_FP8_EC_IX_WGRAD_IDX_AFTER_DOT)
    # vf3p: ops/flydsl/hstu_flydsl_mha.py
    # adds the bwd_dkdv VALU token "vf3p" to the effective
    # DLRM_HSTU_FP8_BWD_VALU set of the dq-split calls (needs ilp; not with
    # vf/vf2/h16/vf3/vf3s; only for 0 < qkv_scale <= 0.25). Read per call;
    # absent or "0" = off = the default act chain. A declared f32
    # reassociation (ulp-level; no format or scale change).
    _bind_bool("DLRM_HSTU_FP8_BWD_VF3P", DLRM_HSTU_FP8_BWD_VF3P)
    # vf4 (DECLARED PRECISION CHANGE, gin binds it True):
    # ops/flydsl/hstu_flydsl_mha.py adds the bwd_dkdv VALU
    # token "vf4" on top of vf3p (6-op SiLU-backward chain; the silu e4m3
    # cast is re-rounded on a pow2-shifted grid, dS ulp-level; one build for
    # every qkv_scale). Read per call; absent or "0" = off = vf3p.
    _bind_bool("DLRM_HSTU_FP8_BWD_VF4", DLRM_HSTU_FP8_BWD_VF4)
    # DSC_K (declared numerics change): the C8 attention
    # backward reads DLRM_HSTU_FP8_BWD_FULL_DSC_K (ops/flydsl/hstu_flydsl_mha.py
    # _sc("DSC", "-3"): ds e4m3 cast scale 2**(ceil(log2 N) + K)). A value, not
    # a boolean; env wins; a non-integer fails here instead of at the first
    # backward.
    _bind_value("DLRM_HSTU_FP8_BWD_FULL_DSC_K", DLRM_HSTU_FP8_BWD_FULL_DSC_K)
    if os.environ.get("DLRM_HSTU_FP8_BWD_FULL_DSC_K", "") != "":
        int(os.environ["DLRM_HSTU_FP8_BWD_FULL_DSC_K"])
    # The targets-only layer's V/K projection reads the step-tiled yT on the
    # K512 library's tiled-A entry, and its Triton v2 store_t is called with
    # store_y=False. The kernel still WRITES y's bytes, but into a dedicated
    # 1/32-size scratch that is discarded (16*M+512 bytes, row stride 16), not
    # into an [M, 512] y: fewer DRAM write-backs, not fewer stores. Bitwise.
    # Read at import of ops/triton/triton_fp8.py (after this runs). Absent =
    # off in code; only "1" arms it; DLRM_HSTU_FP8_FWD_K512_TILED_A=0 in the
    # environment wins over this binding and restores the row-major y and
    # V/K projection.
    _bind_bool("DLRM_HSTU_FP8_FWD_K512_TILED_A", DLRM_HSTU_FP8_FWD_K512_TILED_A)
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
    # Action-branch prefill (modules/action_prefill.py): the input
    # preprocessor's embedding-independent action branch is issued before the
    # forward embedding all-to-all wait. Read per forward; absent = off in
    # code. Bitwise. Composes with the grouped tail above (the prefilled action
    # head joins the group; ops/preproc_grouped_tail.py). Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_ACTION_PREFILL", DLRM_HSTU_FP8_ACTION_PREFILL)
    # Its sub-switch (action_prefill.l2_enabled, read per prefill): the prefill
    # also runs the action MLP's last Linear. Absent = on in code, and the gin
    # binds True: the same "1"; inert while the switch above is off. A pre-set
    # "0" keeps that Linear in place (part A only). Bitwise either way.
    _bind_bool("DLRM_HSTU_FP8_ACTION_PREFILL_L2", DLRM_HSTU_FP8_ACTION_PREFILL_L2)
    # Next-batch input dist waits on its own batch's H2D event, not on the
    # whole memcpy stream (dlrm_v4/train/dd_own_h2d.py). Read when the train
    # pipeline is built; absent = off in code. Bitwise. Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_DD_OWN_H2D", DLRM_HSTU_FP8_DD_OWN_H2D)
    # Eval embedding lookahead (dlrm_v4/train/eval_lookahead.py): the eval
    # pipeline issues batch N+1's EC lookup and sequence all-to-all on a side
    # stream at the end of progress(N). Read before RUN_START (train_ranker)
    # and when the eval pipeline is built; absent = off in code, the gin binds
    # True; only "1" arms it. Bitwise eval. Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_EVAL_EMB_LOOKAHEAD", DLRM_HSTU_FP8_EVAL_EMB_LOOKAHEAD)
    # Train-pipeline memcpy / data-dist stream priority when the job spans
    # more than one node (host_sync_free.train_pipe_stream_priority; read when
    # the pre-grown pairs and the train pipeline are built). 1-node runs and
    # the eval pipeline keep torchrec's -1. Bitwise. Not FP8 math.
    _bind_value(
        "DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN", DLRM_HSTU_FP8_PIPE_STREAM_PRIO_MN
    )
    # Reference-data-only staging: a flag the launcher's data-staging step may
    # read; nothing in the trainer reads the variable. Bound here so the
    # binding is a declared parameter and the trainer's environment records
    # it. Not FP8 math.
    _bind_bool("DLRM_HSTU_FP8_STAGE_REF_ONLY", DLRM_HSTU_FP8_STAGE_REF_ONLY)
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
            "well to run bf16, plus DLRM_HSTU_FP8_PREPROC_L2=0 and "
            "DLRM_HSTU_FP8_BWD_UVQK=0: the gin arms those two FP8 linear paths "
            "independently of FWD/BWD (a non-gfx950 host wants all of them "
            "off, alongside HSTU_ATTN_BWD=triton), or turn the forward back on."
        )
