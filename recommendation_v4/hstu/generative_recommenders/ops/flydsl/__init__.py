"""FlyDSL HSTU attention backward for gfx950 (MI350X).

Deliberately empty of imports: `hstu_bwd_dkdv` pulls in the `flydsl` package at
module scope, which is not a dependency of the trainer. Importing this package
must stay free so a build without flydsl installed can still resolve
`generative_recommenders.ops.flydsl.hstu_flydsl_mha` -- which is how a host that
cannot run the kernel produces a useful error instead of an ImportError.

Hence the shared default lives here rather than in either dispatch site: both
`ops/hstu_attention.py` and `hstu_flydsl_mha.py` need it to interpret an unset
$HSTU_ATTN_BWD, and two copies of the string would eventually disagree.
"""

# Backward kernel used when $HSTU_ATTN_BWD is unset. In a trainer run the gin
# binding (apply_env_bootstrap.HSTU_ATTN_BWD) sets the variable explicitly, so
# this governs direct library use and tests.
DEFAULT_BWD_BACKEND = "flydsl"

# Forward kernel used when $HSTU_ATTN_FWD is unset. The trainer gin default is
# flydsl; this library default stays triton so tests and non-gin callers do
# not raise off gfx950. DLRM_HSTU_FP8_FWD plus HSTU_ATTN_FWD=flydsl selects
# the FlyDSL e4m3 forward. The backward default remains flydsl.
DEFAULT_FWD_BACKEND = "triton"

# Stdlib only, so the package import stays free (see the module docstring).
import logging as _logging  # noqa: E402
import os as _os  # noqa: E402

_logger = _logging.getLogger(__name__)
_stream_logged = set()


def current_stream_launch() -> bool:
    """$DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM (default "1"; gin:
    apply_env_bootstrap.DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM).

    True: the FlyDSL attention fwd/bwd call() wrappers launch on
    torch.cuda.current_stream(q.device) (Arbor iter 6, 5ae3127). False: they
    pass no stream, so launch() uses its default fx.Stream(None) -- HIP's null
    stream, the exact pre-5ae3127 launch arguments. Not an FP8 knob; the
    prefix is only what launch_slurm.sh forwards into the container.

    The host-sync-free step ($DLRM_HSTU_FP8_SYNCFREE=1) requires True: it runs
    FlyDSL while torchrec side streams are current, where the null-stream
    launch raced its producers (20/20 wrong outputs, Arbor iter 6 job 53750).
    Read per call (cheap) because the gin env bootstrap may run after import.
    """
    raw = _os.environ.get("DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM", "1")
    on = raw.strip().lower() not in ("0", "false", "off", "no")
    if not on and _os.environ.get("DLRM_HSTU_FP8_SYNCFREE", "0") == "1":
        raise RuntimeError(
            "DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM=%r with "
            "DLRM_HSTU_FP8_SYNCFREE=1: the host-sync-free step needs FlyDSL "
            "launches on the current stream (the legacy null-stream launch "
            "raced side-stream producers, 20/20 wrong outputs). Set "
            "DLRM_HSTU_FP8_SYNCFREE=0 to test the legacy launch stream." % raw
        )
    if on not in _stream_logged:
        _stream_logged.add(on)
        _logger.warning("[flydsl] launch stream: %s",
                        "current" if on else "legacy")
    return on
