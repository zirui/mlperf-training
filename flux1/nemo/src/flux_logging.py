import logging
from nemo.utils import logging as nemo_logging
import builtins
import warnings

import _log_suppression


class RankZeroFilter(logging.Filter):
    def __init__(self, rank):
        self.rank = rank

    def filter(self, record):
        return self.rank == 0


def get_rank_zero_logger(name, rank):
    logger = logging.getLogger(name)
    logger.addFilter(RankZeroFilter(rank))
    return logger


def disable_print():
    def do_nothing(*args, **kwargs):
        pass

    builtins.print = do_nothing


def setup_logging():
    """Configure Python logging for the training run.

    Under ``MLPERF_VERBOSE_LOGS=1`` we leave framework log levels alone so
    the user sees all the original NeMo / Lightning / Megatron banners. In
    quiet mode (the default) we re-apply the logger-level suppression set
    up by ``_log_suppression`` at import time -- this is necessary because
    ``nemo`` and ``lightning`` reset their own loggers to INFO inside
    their package ``__init__`` modules, which runs after
    ``_log_suppression.install()`` but before ``setup_logging()``.
    """
    if _log_suppression.VERBOSE_LOGS:
        return

    nemo_logging.set_verbosity(logging.ERROR)
    _log_suppression.reapply_quiet_logger_levels()
    warnings.filterwarnings("ignore")
    warnings.filterwarnings(
        "ignore", message=".*Could not find the bitsandbytes CUDA binary.*"
    )
