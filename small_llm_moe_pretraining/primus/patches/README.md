# Remaining out-of-tree patches

This directory used to contain a larger set of patch files that were
applied against Primus, Primus-Turbo and the Megatron-LM submodule of
Primus at Dockerfile build time. Those patches have been moved upstream:

- Primus source patches -> inlined onto the Primus
  ``mlperf_training_6.0/gpt_oss_20b`` branch
  (base commit ``44f780ddf0750e0fb507ae7f965581f6eb530af1`` from
  ``docker.io/rocm/primus:v26.2``).
- Primus-Turbo Triton hybrid-FP8 dtype fix -> inlined onto the Primus-Turbo
  ``mlperf_training_6.0/gpt_oss_20b`` branch
  (base commit ``c79e0f359a0c0952c634f3b48312cd6341b5d843``).
- Megatron-LM source patches -> bundled inside Primus under
  ``primus/backends/megatron/patches/megatron_lm_source/`` with an
  import-time applier
  (``primus.backends.megatron.patches.megatron_lm_source_patches``).

The patches that remain here modify the TransformerEngine source tree
(and its aiter submodule), which is built at Dockerfile time from the
ROCm TransformerEngine fork. They cannot be moved into Primus or
Primus-Turbo because the code they change does not live there.

| Patch                                  | Target                                                                 | Why it cannot move                                                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| ``aiter_fwd_v3_silent_fallback.patch`` | ``TransformerEngine/3rdparty/aiter/csrc/cpp_itfs/mha_fwd.cu``          | aiter is an upstream submodule of the ROCm TransformerEngine fork, not part of Primus or Primus-Turbo.              |
| ``aiter_fused_rope.patch``             | ``TransformerEngine/transformer_engine/pytorch/attention/rope.py``    | Modifies TE's RoPE wrapper to honour ``NVTE_USE_AITER_ROPE`` and surface import errors; no equivalent in Primus.    |
| ``te_fmha_backend_selection.patch``    | ``TransformerEngine/transformer_engine/pytorch/attention/...backends.py`` | Introduces TE-internal env-var plumbing for FMHA backend selection; the logic only makes sense in the TE code path. |
| ``megatron_validation_consumed_samples_v0.15.3.patch`` | ``Megatron-LM`` at tag ``v0.15.3`` (inside ``Dockerfile.nvidia``)       | Used exclusively by ``Dockerfile.nvidia``, which clones upstream ``AMD-AIG-AIMA/Primus`` / Megatron ``v0.15.3`` without our Primus ``mlperf_training_6.0/gpt_oss_20b`` bundled patches. The ROCm Dockerfile no longer needs it. |

Landing any of these upstream in the ROCm TransformerEngine repo (and
then bumping ``TE_COMMIT`` in the Dockerfile) would let us drop them
from this directory. ``megatron_validation_consumed_samples_v0.15.3.patch``
can be dropped once ``Dockerfile.nvidia`` is reworked to consume the
same bundled Megatron-LM source patches shipped inside Primus.
