# Copyright (c) 2025, NVIDIA CORPORATION & AMD. All rights reserved.
#
# Monkey-patches for Megatron's training loop to inject:
#   1. MLPerf eval logging + convergence checking (evaluate_and_print_results)
#   2. MLPerf training loss logging (training_log)
#
# These hooks are installed before pretrain() is called and intercept
# Megatron-internal functions without modifying Megatron source code.
#
# IMPORTANT: The function signatures depend on the Megatron version in
# rocm/primus:v26.2 (megatron-core 0.15.0rc8). If signatures change,
# adjust the wrappers below.
#
# Verified 2026-04-02 inside rocm/mlperf:flux1_training_6.0_2026-04-02-17-25-39:
#
#   evaluate_and_print_results(prefix, forward_step_func, data_iterator,
#       model, iteration[4], process_non_loss_data_func, config,
#       verbose=False, write_to_tensorboard=True, non_loss_data_func=None)
#
#   evaluate(forward_step_func, data_iterator, model,
#       process_non_loss_data_func, config, verbose=False,
#       non_loss_data_func=None, eval_iters=None)
#
#   training_log(loss_dict[0], total_loss_dict, learning_rate[2],
#       decoupled_learning_rate, iteration[4], loss_scale,
#       report_memory_flag, skipped_iter, grad_norm, params_norm,
#       num_zeros_in_grad)
#
# Re-verify with:
#   PYTHONPATH=/workspace/Primus/third_party/Megatron-LM:$PYTHONPATH \
#   python3 -c "import inspect, megatron.training.training as m; \
#     print(inspect.signature(m.evaluate_and_print_results)); \
#     print(inspect.signature(m.evaluate)); \
#     print(inspect.signature(m.training_log))"

import logging

import torch

logger = logging.getLogger(__name__)


def install_eval_hook(mlperf_logger, target_val_loss):
    """
    Patch evaluate_and_print_results (and its inner evaluate call) to:
    1. Log MLPerf validation start/end events
    2. Check convergence and stop training if val_loss <= target

    evaluate_and_print_results returns None, so we also patch evaluate()
    to capture total_loss_dict.
    """
    import megatron.training.training as mt
    from megatron.training import get_args

    _orig = mt.evaluate_and_print_results
    _orig_evaluate = mt.evaluate
    _captured_loss = {}

    def _hooked_evaluate(*args, **kwargs):
        result = _orig_evaluate(*args, **kwargs)
        total_loss_dict = result[0] if isinstance(result, tuple) else result
        if isinstance(total_loss_dict, dict):
            _captured_loss.clear()
            _captured_loss.update(total_loss_dict)
        return result

    mt.evaluate = _hooked_evaluate

    def _hooked_evaluate_and_print_results(*args, **kwargs):
        iteration = _extract_iteration(args, kwargs)
        mlperf_logger.on_validation_start(iteration)
        _captured_loss.clear()

        result = _orig(*args, **kwargs)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        val_loss = _extract_val_loss(_captured_loss)
        if val_loss is not None:
            mlperf_logger.on_validation_end(iteration, val_loss)
            logger.info("Validation loss at step %d: %.6f (target: %.6f)",
                        iteration, val_loss, target_val_loss)

            if val_loss <= target_val_loss:
                logger.info("Convergence reached! val_loss=%.6f <= target=%.6f",
                            val_loss, target_val_loss)
                mlperf_logger.log_run_stop(success=True, global_step=iteration)
                try:
                    get_args().train_iters = iteration
                except Exception:
                    logger.warning("Could not set args.train_iters for early stop")
        else:
            logger.warning("Could not extract validation loss from evaluate result")

        return result

    mt.evaluate_and_print_results = _hooked_evaluate_and_print_results
    logger.info("Installed eval hook on megatron.training.training.evaluate_and_print_results")


def install_training_log_hook(mlperf_logger):
    """
    Patch training_log to call mlperf_logger.on_train_batch_end with loss/LR.

    Megatron calls training_log() every iteration (the log_interval controls
    what gets printed, but the function is always invoked).

    On the first call, we also log INIT_STOP + RUN_START — that is the
    earliest hook point after Megatron has finished internal initialization.
    """
    import megatron.training.training as mt

    _orig_log = mt.training_log
    _first_call = [True]

    def _hooked_training_log(*args, **kwargs):
        if _first_call[0]:
            _first_call[0] = False
            mlperf_logger.log_init_stop_run_start()

        result = _orig_log(*args, **kwargs)

        # training_log signature (upstream):
        #   training_log(loss_dict, total_loss_dict, learning_rate,
        #                decoupled_learning_rate, iteration, loss_scale,
        #                report_memory_flag, skipped_iter, ...)
        try:
            loss_dict = args[0] if len(args) > 0 else kwargs.get("loss_dict", {})
            learning_rate = args[2] if len(args) > 2 else kwargs.get("learning_rate", 0.0)
            iteration = args[4] if len(args) > 4 else kwargs.get("iteration", 0)

            if loss_dict:
                loss_val = next(iter(loss_dict.values()))
                if hasattr(loss_val, "item"):
                    loss_val = loss_val.item()
                mlperf_logger.on_train_batch_end(iteration, loss_val, learning_rate)
        except Exception as e:
            logger.debug("training_log hook: could not extract loss/lr: %s", e)

        return result

    mt.training_log = _hooked_training_log
    logger.info("Installed training_log hook on megatron.training.training.training_log")


def _extract_iteration(args, kwargs):
    """Extract iteration from evaluate_and_print_results args, defensively."""
    if "iteration" in kwargs:
        return kwargs["iteration"]
    # Positional arg index 4 in upstream Megatron
    if len(args) > 4:
        val = args[4]
        if isinstance(val, int):
            return val
    # Fallback: try all int args
    for a in args:
        if isinstance(a, int) and a >= 0:
            return a
    return 0


def _extract_val_loss(loss_dict):
    """
    Extract scalar validation loss from the captured total_loss_dict.

    Uses the "loss" key directly — Megatron's evaluate_and_print_results already
    averages across eval_iters and all-reduces across DP ranks. Doing a second
    all-reduce on loss_sum/sample_count inflates the result.
    """
    if not loss_dict or not isinstance(loss_dict, dict):
        return None

    for key in ("loss", "lm loss"):
        if key in loss_dict:
            val = loss_dict[key]
            return val.item() if hasattr(val, "item") else float(val)

    if loss_dict:
        val = next(iter(loss_dict.values()))
        return val.item() if hasattr(val, "item") else float(val)

    return None
