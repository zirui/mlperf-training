# Copyright (c) 2025, NVIDIA CORPORATION & AMD. All rights reserved.
#
# Energon-based dataset provider for Flux training.
# Replaces NeMo DiffusionDataModule + CustomDiffusionDataModule.
# TaskEncoder ported from flux1/nemo/src/task_encoder.py (already NeMo-free).

import io
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---- Task encoder (from flux1/nemo/src/task_encoder.py) ----

def deserialize_numpy_array(data: bytes) -> torch.Tensor:
    buffer = io.BytesIO(data)
    uint16_data = np.load(buffer)
    tensor = torch.from_numpy(uint16_data).view(torch.bfloat16)
    return tensor


def _import_energon():
    from megatron.core import parallel_state
    from megatron.energon import (
        Cooker,
        CrudeSample,
        DefaultTaskEncoder,
        WorkerConfig,
        basic_sample_keys,
        get_loader,
        get_train_dataset,
    )
    from megatron.energon.task_encoder.base import stateless
    return (parallel_state, Cooker, CrudeSample, DefaultTaskEncoder,
            WorkerConfig, basic_sample_keys, get_loader, get_train_dataset, stateless)


# ---- Mock dataset for smoke testing without real data ----

class MockFluxDataset(Dataset):
    """Map-style dataset producing random tensors shaped like preprocessed Flux data."""

    def __init__(self, num_samples: int, latent_h: int = 32, latent_w: int = 32,
                 context_dim: int = 4096, vec_in_dim: int = 768,
                 seq_len: int = 256, is_validation: bool = False):
        self.num_samples = num_samples
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.context_dim = context_dim
        self.vec_in_dim = vec_in_dim
        self.seq_len = seq_len
        self.is_validation = is_validation

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        logvar_scale = float(os.environ.get("MOCK_LOGVAR_SCALE", "0"))
        mean_scale = float(os.environ.get("MOCK_MEAN_SCALE", "1"))
        sample = {
            "mean": mean_scale * torch.randn(16, self.latent_h, self.latent_w, dtype=torch.bfloat16),
            "logvar": logvar_scale * torch.randn(16, self.latent_h, self.latent_w, dtype=torch.bfloat16),
            "prompt_embeds": torch.randn(self.seq_len, self.context_dim, dtype=torch.bfloat16),
            "pooled_prompt_embeds": torch.randn(self.vec_in_dim, dtype=torch.bfloat16),
        }
        if self.is_validation:
            sample["timestep"] = torch.tensor(idx % 8)
        return sample


# ---- Energon dataset provider ----

def get_energon_datasets(
    data_path: str,
    micro_batch_size: int,
    num_workers: int = 8,
    virtual_epoch_length: int = 1099776,
    packing_buffer_size: int = None,
    use_train_split_for_val: bool = False,
):
    """
    Build train and val datasets using megatron.energon.
    Returns (train_dataset, val_dataset) iterables that yield batches of dicts.
    """
    (parallel_state, Cooker, CrudeSample, DefaultTaskEncoder,
     WorkerConfig, basic_sample_keys, get_loader, get_train_dataset, stateless) = _import_energon()

    @stateless
    def deserialize_preprocessed_example(example):
        sample = {}
        sample["prompt_embeds"] = deserialize_numpy_array(example["t5.bytes"])
        sample["pooled_prompt_embeds"] = deserialize_numpy_array(example["clip.bytes"])
        sample["mean"] = deserialize_numpy_array(example["mean.bytes"])
        sample["logvar"] = deserialize_numpy_array(example["logvar.bytes"])
        if "json" in example:
            decoded = json.loads(example["json"])
            if "timestep" in decoded:
                sample["timestep"] = torch.tensor(decoded["timestep"])
        return CrudeSample(**basic_sample_keys(example), **sample)

    class FluxTaskEncoder(DefaultTaskEncoder):
        decoder = None
        cookers = [Cooker(deserialize_preprocessed_example)]

    worker_config = WorkerConfig(
        rank=parallel_state.get_data_parallel_rank(),
        world_size=parallel_state.get_data_parallel_world_size(),
        num_workers=num_workers,
        data_parallel_group=parallel_state.get_data_parallel_group(),
    )

    task_encoder = FluxTaskEncoder()

    train_ds = get_train_dataset(
        data_path,
        batch_size=micro_batch_size,
        task_encoder=task_encoder,
        worker_config=worker_config,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        split_part="train",
        virtual_epoch_length=virtual_epoch_length,
        packing_buffer_size=packing_buffer_size,
    )

    val_split = "train" if use_train_split_for_val else "val"
    val_ds = get_train_dataset(
        data_path,
        batch_size=micro_batch_size,
        task_encoder=task_encoder,
        worker_config=worker_config,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        split_part=val_split,
        virtual_epoch_length=0,
        packing_buffer_size=packing_buffer_size,
    )

    train_loader = get_loader(train_ds)
    val_loader = get_loader(val_ds)

    return train_loader, val_loader


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """
    Megatron pretrain()-compatible dataset provider signature.
    When MOCK_DATA=true, returns MockFluxDataset (random tensors) for smoke testing.
    Otherwise, loads real data from DATA_PATH via megatron.energon and wraps with
    get_loader() to provide the required worker context for iteration.
    """
    mock_data = os.environ.get("MOCK_DATA", "false").lower() == "true"

    if mock_data:
        latent_h = int(os.environ.get("MOCK_LATENT_H", "32"))
        latent_w = int(os.environ.get("MOCK_LATENT_W", "32"))
        logger.info("Using mock data for smoke testing (latent %dx%d)", latent_h, latent_w)
        train_ds = MockFluxDataset(num_samples=10000, latent_h=latent_h, latent_w=latent_w, is_validation=False)
        eval_iters = int(os.environ.get("EVAL_ITERS", "64"))
        val_ds = MockFluxDataset(num_samples=max(256, eval_iters * 16), latent_h=latent_h, latent_w=latent_w, is_validation=True) if eval_iters > 0 else None
        return train_ds, val_ds, None

    data_path = os.environ.get("DATA_PATH", "/dataset/energon")
    mbs = int(os.environ.get("MBS", "1"))
    num_workers = int(os.environ.get("NUM_WORKERS", "8"))

    train_loader, val_loader = get_energon_datasets(
        data_path=data_path,
        micro_batch_size=mbs,
        num_workers=num_workers,
    )
    return train_loader, val_loader, None
