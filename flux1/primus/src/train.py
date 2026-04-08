# Copyright (c) 2025, NVIDIA CORPORATION & AMD. All rights reserved.
#
# Entry point for Flux training using Megatron pretrain() -- NeMo-free.
# Config is loaded from the YAML file specified by the EXP env var;
# environment variables can override any YAML value via ${VAR:default}.
#
# Usage:
#   torchrun --nproc_per_node=8 src/train.py

import gc
import logging
import os

import torch
import torch.utils.data
import megatron.training.initialize as _megatron_init
from megatron.core.transformer.enums import ModelType
from megatron.training import training as _mt_training
from megatron.training.training import pretrain
from torch import nn

# ROCm: skip legacy CUDA fused-kernel compilation (nvcc not available).
_megatron_init._compile_dependencies = lambda: None

# NeMo sets check_for_nan_in_grad=False on its DDP config. Megatron's
# DistributedDataParallelConfig defaults to True, which triggers the
# rerun_state_machine on transient NaN grad norms and crashes.
from megatron.core.distributed import DistributedDataParallelConfig as _DDPConfig
_orig_ddp_post_init = getattr(_DDPConfig, '__post_init__', None)
def _patched_ddp_post_init(self):
    if _orig_ddp_post_init:
        _orig_ddp_post_init(self)
    self.check_for_nan_in_grad = False
_DDPConfig.__post_init__ = _patched_ddp_post_init

# Monkey-patch build_pretraining_data_loader to handle energon loaders.
_orig_build_loader = _mt_training.build_pretraining_data_loader

def _patched_build_pretraining_data_loader(dataset, *args, **kwargs):
    if dataset is None:
        return None
    if isinstance(dataset, (torch.utils.data.DataLoader, torch.utils.data.IterableDataset)):
        return dataset
    return _orig_build_loader(dataset, *args, **kwargs)

_mt_training.build_pretraining_data_loader = _patched_build_pretraining_data_loader

from config_loader import load_flux_config
from flux_model import CustomFluxConfig
from flux_forward_step import FluxForwardContext, make_forward_step_func
from flux_data import train_valid_test_datasets_provider
from mlperf_callbacks import FluxMLPerfLogger
from megatron_hooks import install_eval_hook, install_training_log_hook
from warmup import install_fp8_warmup_hook
from nan_detect import install_training_nan_hooks


logger = logging.getLogger(__name__)

# Global config dict populated from YAML in main(), read by model_provider
_CFG = {}


def flux_model_provider(pre_process=True, post_process=True):
    """Megatron model_provider signature -- returns a Flux model instance."""
    from megatron.training import get_args
    args = get_args()
    cfg = _CFG

    fp8_mode = cfg.get("fp8")
    if isinstance(fp8_mode, str) and fp8_mode.lower() == "none":
        fp8_mode = None
    if fp8_mode is None:
        logger.warning("FP8 is DISABLED. Set fp8: hybrid in YAML to match NeMo.")

    config = CustomFluxConfig(
        num_joint_layers=cfg.get("num_joint_layers", 19),
        num_single_layers=cfg.get("num_single_layers", 38),
        hidden_size=cfg.get("hidden_size", 3072),
        num_attention_heads=cfg.get("num_attention_heads", 24),
        in_channels=cfg.get("in_channels", 64),
        context_dim=cfg.get("context_dim", 4096),
        model_channels=cfg.get("model_channels", 256),
        vec_in_dim=cfg.get("vec_in_dim", 768),
        guidance_embed=cfg.get("guidance_embed", False),
        calculate_per_token_loss=False,
        gradient_accumulation_fusion=cfg.get("gradient_accumulation_fusion", False),
        enable_cuda_graph=cfg.get("use_cuda_graph", False),
        cuda_graph_scope=cfg.get("cuda_graph_scope", "full_iteration"),
        use_te_rng_tracker=True,
        cuda_graph_warmup_steps=2,
        apply_rope_fusion=cfg.get("apply_rope_fusion", True),
        rotary_interleaved=True,
        cpu_offloading=cfg.get("cpu_offloading", True),
        cpu_offloading_activations=cfg.get("cpu_offloading_activations", True),
        cpu_offloading_weights=cfg.get("cpu_offloading_weights", False),
        init_method=nn.init.xavier_uniform_,
        output_layer_init_method=nn.init.xavier_uniform_,
        fp8=fp8_mode,
        fp8_recipe=cfg.get("fp8_recipe", "delayed"),
        fp8_margin=cfg.get("fp8_margin", 0),
        fp8_amax_history_len=cfg.get("fp8_amax_history_len", 1024),
        fp8_amax_compute_algo=cfg.get("fp8_amax_compute_algo", "max"),
        ckpt_path=cfg.get("load"),
        load_dist_ckpt=cfg.get("load_dist_ckpt", False),
        bf16=args.bf16,
        fp16=args.fp16,
        params_dtype=args.params_dtype,
    )

    print(f"[Primus] FluxConfig: fp8={config.fp8}, normalization={config.normalization}, "
          f"hidden_size={config.hidden_size}, num_joint_layers={config.num_joint_layers}, "
          f"num_single_layers={config.num_single_layers}, bf16={config.bf16}, "
          f"params_dtype={config.params_dtype}", flush=True)

    model = config.configure_model()

    return model


def add_flux_args(parser):
    """Add Flux-specific arguments to Megatron argument parser."""
    group = parser.add_argument_group("Flux", "Flux training arguments")
    group.add_argument("--empty-encodings-path", type=str,
                       default="/dataset/energon/empty_encodings")
    group.add_argument("--classifier-free-guidance-prob", type=float,
                       default=0.1)
    group.add_argument("--target-accuracy", type=float,
                       default=0.586)
    group.add_argument("--generate-random-empty-encodings", action="store_true",
                       default=False)
    return parser


def _build_megatron_cli_args(cfg):
    """Build Megatron CLI args list from the YAML config dict."""
    args = []

    if cfg.get("use_distributed_optimizer", True):
        args.append("--use-distributed-optimizer")
    if cfg.get("overlap_grad_reduce", True):
        args.append("--overlap-grad-reduce")

    args.append("--no-check-for-nan-in-loss-and-grad")

    _map = {
        "ddp_bucket_size": "--ddp-bucket-size",
        "lr": "--lr",
        "lr_warmup_iters": "--lr-warmup-iters",
        "weight_decay": "--weight-decay",
        "adam_beta1": "--adam-beta1",
        "adam_beta2": "--adam-beta2",
        "adam_eps": "--adam-eps",
        "clip_grad": "--clip-grad",
        "micro_batch_size": "--micro-batch-size",
        "global_batch_size": "--global-batch-size",
        "train_iters": "--train-iters",
        "eval_iters": "--eval-iters",
        "eval_interval": "--eval-interval",
        "log_interval": "--log-interval",
        "seed": "--seed",
    }

    for yaml_key, cli_flag in _map.items():
        val = cfg.get(yaml_key)
        if val is not None:
            args.extend([cli_flag, str(val)])

    lr_decay = cfg.get("lr_decay_style", "constant")
    args.extend(["--lr-decay-style", lr_decay])

    dl_type = cfg.get("dataloader_type", "cyclic")
    args.extend(["--dataloader-type", dl_type])

    return args


def main():
    global _CFG

    config_path = os.environ.get("EXP", "/workspace/code/conf/flux1-pretrain-fp8.yaml")
    _CFG = load_flux_config(config_path)
    cfg = _CFG

    print(f"[Primus] Loaded config from: {config_path}", flush=True)

    seed = cfg.get("seed", 2025)
    torch.manual_seed(seed)

    gbs = cfg.get("global_batch_size", 512)
    mbs = cfg.get("micro_batch_size", 64)
    target = cfg.get("target_val_loss", 0.586)
    nnodes = int(os.environ.get("NNODES", os.environ.get("DGXNNODES", "1")))

    mlperf_logger = FluxMLPerfLogger(
        global_batch_size=gbs,
        micro_batch_size=mbs,
        target_accuracy=target,
    )
    mlperf_logger.log_init(num_nodes=nnodes, seed=seed)
    mlperf_logger.log_hyperparams(config=None)

    install_eval_hook(mlperf_logger, target)
    install_training_log_hook(mlperf_logger)

    forward_ctx = FluxForwardContext(
        empty_encodings_path=cfg.get("empty_encodings_path", "/dataset/energon/empty_encodings"),
        context_dim=cfg.get("context_dim", 4096),
        vec_in_dim=cfg.get("vec_in_dim", 768),
        classifier_free_guidance_prob=cfg.get("classifier_free_guidance_prob", 0.1),
        autocast_dtype=torch.bfloat16,
        generate_random_empty_encodings=cfg.get("generate_random_empty_encodings", False),
    )
    forward_step_func = make_forward_step_func(forward_ctx)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.recompile_limit = 64

    install_fp8_warmup_hook()

    nan_detect = cfg.get("nan_detect", 1)
    if str(nan_detect) == "1" or nan_detect is True:
        install_training_nan_hooks()

    gc.disable()

    # Build Megatron CLI args from the YAML config
    megatron_args = _build_megatron_cli_args(cfg)
    import sys
    sys.argv = [sys.argv[0]] + megatron_args

    pretrain(
        train_valid_test_datasets_provider,
        flux_model_provider,
        ModelType.encoder_or_decoder,
        forward_step_func,
        args_defaults={
            "tokenizer_type": "NullTokenizer",
            "vocab_size": 0,
            "num_layers": 57,
            "hidden_size": cfg.get("hidden_size", 3072),
            "num_attention_heads": cfg.get("num_attention_heads", 24),
            "max_position_embeddings": 8192,
            "seq_length": 4096,
            "dataloader_type": cfg.get("dataloader_type", "cyclic"),
        },
        extra_args_provider=add_flux_args,
    )

    if not mlperf_logger.converged:
        mlperf_logger.log_run_stop(success=False, global_step=-1)

    mlperf_logger.teardown()
    gc.enable()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
