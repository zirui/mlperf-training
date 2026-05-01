# ---------------------------------------------------------------------------
# Non-MLLOG log suppression for the Flux1 benchmark.
#
# The MLPerf reference logs (":::MLLOG ...") plus the run banners emitted by
# ``run_and_time.sh`` are the only lines we want on stdout. All other
# framework output (NeMo warnings, Lightning banner, Hydra config dump,
# OneLogger telemetry, Megatron deprecations, multistorageclient warnings,
# ``Fused optimized group norm has not been installed``, hipblaslt
# ``Warning: Latency not found`` C++ noise, hipify, aiter / gloo C++ banners,
# ...) is suppressed by default to keep run logs clean.
#
# Control:
#   MLPERF_VERBOSE_LOGS=1  -> restore the full verbose output
#   MLPERF_VERBOSE_LOGS=0  -> quiet mode (default): only MLLOG + training
#                            timing/result lines are emitted.
#
# Strategy (strongly preferred over stdout scraping):
#   1. Export env vars that the noisy libraries honour (AITER_LOG_LEVEL,
#      NEMO_TESTING, TRANSFORMERS_VERBOSITY, PYTHONWARNINGS, ...).
#   2. Raise Python ``logging`` levels for the actual logger names emitting
#      the noise (``nemo_logger``, ``lightning_utilities.core.rank_zero``,
#      ``custom_flux``, ``custom_callbacks``, ``multistorageclient.config``,
#      ``nemo.lightning.pytorch.strategies.megatron_strategy``, the root
#      logger, ...). Lightning / NeMo reset their own levels inside their
#      package ``__init__`` so callers must re-apply via
#      ``reapply_quiet_logger_levels`` AFTER those imports.
#   3. Pre-instantiate NeMo's logger and raise its verbosity so
#      ``[NeMo W ...]`` warnings fired during later
#      ``from nemo.collections import ...`` imports are dropped.
#
# A handful of sources (unconditional ``std::cout`` in libhipblaslt / aiter /
# gloo C++, raw ``print()`` in installed library code we do not own) cannot
# be controlled via the above. For those a narrow FD-level line filter is
# installed on stdout so even native writes get matched. Stderr carries no
# useful signal in quiet mode (only hipify's ``Successfully preprocessed``
# banner fires there), so FD 2 is redirected wholesale to ``/dev/null``
# instead of being piped and filtered -- if you need stderr back to debug a
# crash, re-run with ``MLPERF_VERBOSE_LOGS=1``.
#
# This module must be imported BEFORE any other import that may log/print at
# import time (TE, NeMo, Lightning, aiter, ...).
# ---------------------------------------------------------------------------
import os as _os
import logging as _logging

VERBOSE_LOGS = _os.environ.get("MLPERF_VERBOSE_LOGS", "0") == "1"

# Logger names that emit non-MLLOG noise. Keep in sync with
# ``reapply_quiet_logger_levels`` (which runs again after framework imports
# reset these levels).
QUIET_LOGGER_NAMES = (
    # NeMo ``[NeMo W ...]`` / ``[NeMo I ...]`` banners -- NeMo instantiates
    # its singleton logger as ``nemo_logger`` (see nemo.utils.nemo_logging).
    "nemo",
    "nemo_logger",
    "nemo.lightning.pytorch.strategies.megatron_strategy",
    # Lightning / Fabric trainer banners, DDP banner, LOCAL_RANK lines and
    # the ``GPU available: ... / TPU available: ...`` startup banner. The
    # Lightning ``rank_zero_info`` / ``rank_zero_warn`` helpers live in
    # ``lightning_utilities.core.rank_zero``, not in any ``lightning.*``
    # module.
    "lightning",
    "lightning.pytorch",
    "lightning.fabric",
    "lightning_utilities",
    "lightning_utilities.core.rank_zero",
    "lightning.pytorch.callbacks.model_summary",
    "lightning.pytorch.utilities.model_summary",
    "lightning.pytorch.accelerators.cuda",
    "pytorch_lightning",
    # OneLogger telemetry banners ("OneLogger: ..." / "No exporters ...").
    "nv_one_logger",
    # Megatron experimental-API deprecation warnings
    # (``fused_indices_to_multihot has reached end of life``).
    "megatron",
    "megatron.core.utils",
    "megatron.core.rerun_state_machine",
    # TransformerEngine loggers.
    "transformer_engine",
    # Multi-storage client chatter:
    # ``multistorageclient.config WARNING - The profile name 'default' is
    # deprecated`` (fires 8x per run).
    "multistorageclient",
    "multistorageclient.config",
    # torch elastic / torchrun ``W0417 ... Setting OMP_NUM_THREADS ...``
    # advisories.
    "torch.distributed",
    "torch.distributed.run",
    "torch.distributed.elastic",
    "torch.distributed.elastic.multiprocessing",
    "torch.distributed.launcher.api",
    # Flux-local loggers used by custom_flux / custom_callbacks / warmup for
    # ``[DATE][custom_flux][INFO] - COMPILE_DIT enabled: ...`` etc.
    "custom_flux",
    "custom_callbacks",
    "warmup",
)


def reapply_quiet_logger_levels() -> None:
    """Raise levels on noisy Python loggers. Safe to call multiple times.

    This is called once near the top of the consumer file (before framework
    imports) and again after those imports, because packages like
    ``lightning`` and ``nemo`` reset their own logger levels to INFO inside
    their package ``__init__`` and would otherwise overwrite the earlier
    call. The root logger is also locked down since a number of Flux modules
    (``custom_flux.py`` in particular) call ``logging.info(...)`` without
    first binding a named logger.
    """
    _logging.getLogger().setLevel(_logging.ERROR)
    for _logger_name in QUIET_LOGGER_NAMES:
        _logging.getLogger(_logger_name).setLevel(_logging.ERROR)


def _configure_non_mllog_logs_quiet() -> None:
    """Silence every non-MLLOG log source we can control via env vars or
    the standard ``logging`` module. :::MLLOG output is untouched because
    MLLogger has its own logger (``mllog_default``) with ``propagate=False``
    and its own stdout handler, so raising other loggers' levels does not
    affect it.
    """
    _os.environ.setdefault("AITER_LOG_LEVEL", "ERROR")
    _os.environ.setdefault("AITER_LOG_MORE", "0")
    _os.environ.setdefault("NEMO_TESTING", "0")
    _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    _os.environ.setdefault("PYTHONWARNINGS", "ignore")

    import warnings as _warnings

    _warnings.filterwarnings("ignore")

    reapply_quiet_logger_levels()

    # Pre-instantiate NeMo's logger so warnings that fire during the
    # ``from nemo.collections import ...`` import chain (e.g.
    # ``[NeMo W ...] The deploy/evaluate module could not be imported``) are
    # emitted at ERROR or above.
    try:
        from nemo.utils import logging as _nemo_logging  # type: ignore

        _nemo_logging.set_verbosity(_logging.ERROR)
    except Exception:
        pass


def _install_fd_level_fallback_filter() -> None:
    """Last-resort line filter for logs that bypass Python ``logging``
    entirely: unconditional ``std::cout`` writes in libhipblaslt / aiter /
    gloo C++ code, and raw ``print()`` statements that live inside
    installed-library source we do not own (``blocks.py`` in the diffusion
    VAE module). Runs at the OS file-descriptor level so native writes to
    FD 1 are also matched.

    Design mirrors the Llama3 suppression module:
      * FD 2 (stderr) is redirected straight to ``/dev/null``. In quiet
        mode the only stderr noise observed in practice is the hipify
        ``"Successfully preprocessed all matching files."`` banner -- there
        is no MLLOG or training signal on stderr, so dropping the stream
        wholesale is cheaper than piping+filtering it. Users who need
        stderr back (e.g. to debug a crash) can re-run with
        ``MLPERF_VERBOSE_LOGS=1``.
      * FD 1 (stdout) is routed through an ``os.pipe`` + reader thread.
        The thread runs each line through the suppression regexes and
        forwards surviving lines to the saved original stdout FD.
      * ``:::MLLOG`` lines always pass through.
    """
    import re
    import sys
    import threading

    ansi_re = re.compile(r"\x1b\[[0-9;]*[ -/]*[@-~]")

    # Only patterns for log sources that neither ``logging.setLevel`` nor any
    # documented env var can silence belong here. All are native writes that
    # reach FD 1 without going through Python, or raw ``print()`` in
    # installed-library source we cannot edit at runtime without a rebuild.
    suppress_patterns = tuple(
        re.compile(p)
        for p in (
            # hipify print() during runtime JIT of TE / aiter / Apex
            # extensions. The message appears on stderr in the main
            # training process (and is dropped by the FD 2 -> /dev/null
            # redirect below) but on stdout for some ranks, so we also
            # catch it here.
            r"^(?:\x1b\[[0-9;]*m)?Successfully preprocessed all matching files\.",
            # libhipblaslt (ROCm) unconditional std::cout: interleaved
            # "Warning: Latency not found for MI_M=..., mi_input_type=...
            # Returning latency value of 32 (really slow)." across 8 ranks.
            # The banner fragments line-wrap unpredictably, so match on any
            # of the stable substrings.
            r"Warning: Latency not found for MI_M=",
            r"Returning latency value of 32 \(really slow\)",
            r"^(?:\s*(?:BFloat8Float8_fnuz|Float8_fnuz|,\s*(?:MI_[MNK]|mi_input_type)=\d*|\d+)\s*)+\.?\s*$",
            # aiter JIT module-load banner printed on first import.
            r"^\[aiter\] import \[",
            # Gloo C++ peer-connect banner from libtorch_cpu.so.
            r"\[Gloo\] Rank \d+ is connected to ",
            r"^Expected number of connected peer ranks is\s*:",
            # aiter C++ hipModuleLoad / hipModuleGetFunction banners
            # (printed across multiple writes, so we also drop orphan
            # " Success" / bare-number continuations that result from
            # interleaved writes from multiple ranks).
            r"\[aiter\] hipModuleLoad: ",
            r"\[aiter\] hipModuleGetFunction: ",
            r"^\s*Success\s*$",
            r"^\s*\d+\s*$",
            # Raw print() from nemo.collections.diffusion.vae.blocks when
            # the apex fused group norm is not installed (which is always
            # on ROCm). Fires once per rank at import time. A .patch file
            # in ``flux1/nemo/patches/`` converts this to a ``logger.info``
            # call on the next image rebuild, but we filter here as well
            # so the currently-shipping image also produces clean output.
            r"^Fused optimized group norm has not been installed\.",
            # torch elastic advisory: ``W0417 ...
            # torch/distributed/run.py:803] Setting OMP_NUM_THREADS ...``
            # and its separator lines. Formatted by torch's custom
            # formatter so it is not suppressed by
            # ``logging.getLogger("torch.distributed.run").setLevel(ERROR)``
            # on all torch versions, hence the regex fallback.
            r"^W\d{4} \d{2}:\d{2}:\d{2}\.\d+ \d+ torch/distributed/run\.py:",
            r"^\s*\*{5,}\s*$",
        )
    )

    def _should_suppress(line: str) -> bool:
        stripped = ansi_re.sub("", line)
        if ":::MLLOG" in stripped:
            return False
        for pat in suppress_patterns:
            if pat.search(stripped):
                return True
        return False

    def _start_reader(read_fd: int, out_fd: int) -> None:
        def _run() -> None:
            buf = b""
            try:
                while True:
                    chunk = _os.read(read_fd, 4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = raw.decode("utf-8", errors="replace")
                        if not _should_suppress(line):
                            _os.write(out_fd, raw + b"\n")
            except Exception:
                # Never let the filter thread bring down the training run.
                pass
            finally:
                if buf:
                    line = buf.decode("utf-8", errors="replace")
                    if not _should_suppress(line):
                        try:
                            _os.write(out_fd, buf)
                        except Exception:
                            pass

        threading.Thread(target=_run, daemon=True).start()

    # Flush any buffered Python stdout/stderr output before we steal the FDs.
    sys.stdout.flush()
    sys.stderr.flush()

    orig_stdout_fd = _os.dup(1)

    # Drop stderr unconditionally: nothing useful lands on FD 2 in quiet mode
    # and the one noise source (hipify) goes away without any regex work.
    devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(devnull_fd, 2)
    _os.close(devnull_fd)

    stdout_r, stdout_w = _os.pipe()
    _os.dup2(stdout_w, 1)
    _os.close(stdout_w)

    _start_reader(stdout_r, orig_stdout_fd)

    sys.stdout = _os.fdopen(1, "w", buffering=1, closefd=False)
    sys.stderr = _os.fdopen(2, "w", buffering=1, closefd=False)


_INSTALLED = False


def install() -> None:
    """Install the quiet-mode suppression exactly once per process.

    Calling this a second time is a no-op.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    if VERBOSE_LOGS:
        return
    _configure_non_mllog_logs_quiet()
    _install_fd_level_fallback_filter()


# Auto-install on import.
install()
