# pyre-strict
"""F1: fold a Linear's bias gradient into the norm backward that follows it.

WHAT IS BEING REMOVED. ContextualPreprocessor builds three MLPs, each
Sequential(Linear, SwishLayerNorm, Linear, LayerNorm) -- six (Linear, Norm)
pairs per step. For each pair autograd computes the Linear's bias gradient as a
standalone aten::sum over the norm's dx. Measured on trace_ctl2 over 8 ranks x
5 steps, per rank per step:

    aten::sum [M, 512]   2.3425 ms    (the LayerNorm(512) side)
    aten::sum [M, 256]   1.6183 ms    (the SwishLayerNorm(256) side)
                         ------
                         3.9608 ms

That sum re-reads the entire [N, D] dx from HBM immediately after the norm
backward has finished writing it. The column sum it wants is available for free
while dx is still in registers, so the fused path computes it there and the
reread is deleted outright. Both sides are HBM-bound, which is why the removal
is worth more than the addition: the standalone sums run at 3.07 TB/s (D=512)
and 2.22 TB/s (D=256) against the part's 5.84 TB/s, and a reduction that never
touches HBM beats one that does at any bandwidth.

WHAT IT COSTS. Measured directly by _f1_bench.py at the in-situ N=2358539,
not estimated: the SUM_DX accumulator adds ~0.83 ms/rank/step across both
sites at the shipped BLOCK_N=8 config, for a NET of +3.1314 ms/rank/step =
+1.079% of the 290.195 ms step. No register spills at either shipped config;
the D=256 swish kernel does go 180 -> 259 VGPRs (2 -> 1 wave/SIMD), which is
already priced into that measured net.

WHY A Sequential SUBCLASS AND NOT A NEW MODULE. The children, their order, and
their attribute names are unchanged, so state_dict keys stay exactly
"0.weight", "0.bias", "1.weight", ... A checkpoint written by the shipped
container loads into this one and vice versa. Replacing the container with a
purpose-built module would have renamed every key and silently broken resume.

WHY THE PAIRS ARE FUSED AT forward() AND NOT BY REWRITING THE NORM. The norm
modules are shared with the rest of the model; the fold is only valid where a
Linear's bias immediately precedes the norm, which is a property of THIS
container, not of the norm. Keeping the change here means the shipped
LayerNorm/SwishLayerNorm and the 3-tuple triton_weighted_layer_norm_bwd -- and
its six external unpack sites -- are untouched.

NUMERICS. dxsum reduces the fp32 dx still in registers, where aten::sum reduces
the bf16 dx after it was stored. This is the MORE accurate of the two, but it
is not bit-identical: measured relative difference 1.6-2.1e-3, the bf16
rounding scale. That is why this arm owes a convergence gate and not merely a
timing gate.
"""

import logging
from typing import List, Optional, Tuple

import torch
import triton

from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm
from generative_recommenders.ops.triton.triton_layer_norm import (
    _weighted_layer_norm_fwd,
    compute_BLOCK_D,
    triton_weighted_layer_norm_bwd_dxsum,
    triton_weighted_layer_norm_fwd,
)

logger: logging.Logger = logging.getLogger(__name__)


def _norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    is_swish: bool,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The shipped norm forward, reused rather than reimplemented.

    The non-swish side is literally triton_weighted_layer_norm_fwd, the same
    call LayerNormFunction.forward makes. The swish side has no such wrapper --
    SwishLayerNormFunction.forward launches _weighted_layer_norm_fwd inline --
    so that launch is mirrored here argument for argument. Both must stay
    identical to the shipped forward, because F1 is a BACKWARD-only change and
    any forward difference would make the A/B unattributable.

    F2 adds the optional `residual`. When it is given the result is
    ``LN(x) + residual`` computed in ONE pass, replacing a separate [M, D]
    read-modify-write against HBM. Note the residual case must bypass
    ``triton_weighted_layer_norm_fwd`` even on the non-swish side: that wrapper
    is a registered torch.library custom op, so extending its schema would
    change the op signature for every existing caller. Launching the kernel
    inline here instead leaves the shipped wrapper untouched, which is also why
    the ADD_RESIDUAL=False path below still routes through it unchanged --
    the control arm must stay byte-identical to today.
    """
    if not is_swish and residual is None:
        return triton_weighted_layer_norm_fwd(x=x, weight=weight, bias=bias, eps=eps)

    N, D = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd
    BLOCK_D = triton.next_power_of_2(D)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
    # pyre-ignore[28]
    _weighted_layer_norm_fwd[grid](
        x, y, y, weight, bias, mean, rstd, N, D, eps,
        x.stride(0), y.stride(0), y.stride(0), 1.0,
        residual, 0 if residual is None else residual.stride(0),
        IS_SWISH=is_swish, TRAINING=True, BLOCK_D=BLOCK_D,
        COMPUTE_MEAN_AND_RSTD=True, OUTPUT_FP8=False, OUTPUT_BOTH=False,
        ADD_RESIDUAL=residual is not None,
    )
    return y, mean, rstd


class _LinearNormFunction(torch.autograd.Function):
    """y = Norm(x @ lin_w.T + lin_b), with lin_b's gradient folded into Norm's bwd.

    lin_w and lin_b are Function INPUTS, not closed-over attributes, so autograd
    routes their gradients to the leaf Parameters' .grad exactly as
    AddmmBackward0 does. Closing over them instead would silently drop the
    Linear's gradients and train a frozen layer -- which looks like a
    convergence bug, not a plumbing bug, and would be found late.
    """

    @staticmethod
    # pyre-ignore[14]
    def forward(  # pyre-ignore[3]
        ctx,
        x: torch.Tensor,
        lin_w: torch.Tensor,
        lin_b: torch.Tensor,
        norm_w: torch.Tensor,
        norm_b: torch.Tensor,
        eps: float,
        is_swish: bool,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.contiguous()
        # AUTOCAST. The shipped nn.Sequential runs its Linear under autocast, so
        # F.linear's operands are cast to the autocast dtype and it is the CAST
        # weight that AddmmBackward0 saves and later uses for dgrad. A custom
        # Function is handed the RAW arguments instead, and backward() runs with
        # autocast DISABLED -- so saving the fp32 Parameter here and meeting it
        # with a bf16 dh in backward is a mismatch ATen refuses outright:
        #     expected mat1 and mat2 to have the same dtype: BFloat16 != float
        # That is not hypothetical; it killed arm f1_r1. The forward survived
        # because autocast fixed up addmm for us, so the failure surfaced only
        # in backward, one full training step later than the cause.
        #
        # Casting HERE rather than in backward is what makes this both legal and
        # numerically identical to the path it replaces: the forward matmul then
        # runs at exactly the precision the shipped path used, and backward
        # reuses those same cast tensors instead of inventing a new precision.
        #
        # norm_w/norm_b are deliberately NOT cast. The Triton norm loads them
        # and promotes to fp32 internally, exactly as it does for the shipped
        # module, and casting them here would change the norm's numerics rather
        # than preserve them.
        dev = x.device.type
        if torch.is_autocast_enabled(dev):
            amp_dtype = torch.get_autocast_dtype(dev)
            x_c = x.to(amp_dtype)
            lin_w_c = lin_w.to(amp_dtype)
            lin_b_c = lin_b.to(amp_dtype)
        else:
            x_c, lin_w_c, lin_b_c = x, lin_w, lin_b
        h = torch.addmm(lin_b_c, x_c, lin_w_c.t())
        # F2. `residual` is added INSIDE the norm's epilogue instead of by a
        # separate [M, D] read-modify-write against HBM immediately afterwards.
        #
        # The dtype is checked rather than coerced. The shipped `+` this
        # replaces is a plain aten::add, so a mismatch there would type-promote;
        # the kernel cannot, it rounds to Y's dtype before adding. Silently
        # casting would make the fused path disagree with the path it claims to
        # reproduce, and the disagreement would show up as a convergence
        # difference nobody had budgeted for. At the only call site both
        # operands descend from the same autocast region, so this raises only if
        # that stops being true -- which is exactly when someone should be told.
        if residual is not None:
            if residual.dtype != h.dtype:
                raise RuntimeError(
                    "fuse_linear_norm: residual dtype "
                    f"{residual.dtype} != norm output dtype {h.dtype}; the "
                    "fused epilogue rounds to the output dtype before adding "
                    "and cannot reproduce aten::add's type promotion"
                )
            if residual.shape != h.shape:
                raise RuntimeError(
                    f"fuse_linear_norm: residual shape {tuple(residual.shape)} "
                    f"!= norm output shape {tuple(h.shape)}"
                )
            residual = residual.contiguous()
        y, mean, rstd = _norm_fwd(h, norm_w, norm_b, eps, is_swish, residual)
        # h is saved by the shipped path too -- it is the norm's own input, and
        # x is saved for the Linear's wgrad. So this ctx holds no more
        # activation memory than Addmm + Norm already held separately.
        ctx.save_for_backward(x_c, lin_w_c, lin_b_c, h, norm_w, norm_b, mean, rstd)
        # Gradients must be returned in the dtype of the ORIGINAL inputs, not of
        # the cast copies -- AccumulateGrad writes into the fp32 Parameter.
        ctx.grad_dtypes = (x.dtype, lin_w.dtype, lin_b.dtype)
        # F2's residual is not saved. Its gradient is the incoming dy unchanged
        # (d/dr of LN(h) + r is the identity), so backward needs the dtype and
        # nothing else -- no activation memory, no saved tensor.
        ctx.residual_dtype = None if residual is None else residual.dtype
        ctx.eps = eps
        ctx.is_swish = is_swish
        ctx.BLOCK_D = compute_BLOCK_D(h)
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(  # pyre-ignore[3]
        ctx, dy: torch.Tensor
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor], None, None,
        Optional[torch.Tensor],
    ]:
        x, lin_w, lin_b, h, norm_w, norm_b, mean, rstd = ctx.saved_tensors
        dh, dnorm_w, dnorm_b, dh_colsum = triton_weighted_layer_norm_bwd_dxsum(
            dy=dy.contiguous(),
            x=h,
            weight=norm_w,
            bias=norm_b,
            mean=mean,
            rstd=rstd,
            eps=ctx.eps,
            BLOCK_D=ctx.BLOCK_D,
            is_swish=ctx.is_swish,
        )
        needs = ctx.needs_input_grad
        dx = dh @ lin_w if needs[0] else None
        dlin_w = dh.t() @ x if needs[1] else None
        # THE POINT OF THE WHOLE EXERCISE: the Linear's bias gradient comes from
        # the norm backward's register-resident reduction, so the standalone
        # aten::sum over [N, D] that autograd would have emitted never runs.
        dlin_b = dh_colsum if needs[2] else None

        # Grads must come back in the dtype of the INPUT they belong to. dxsum
        # is fp32 by construction and the matmul results follow dh, so under a
        # bf16 parameter dtype these would otherwise mismatch at AccumulateGrad.
        #
        # For x/lin_w/lin_b the reference is ctx.grad_dtypes -- the dtypes the
        # CALLER passed in -- not the saved tensors, which under autocast are
        # the bf16 casts. Referencing the saved copies would hand AccumulateGrad
        # a bf16 gradient for an fp32 Parameter.
        x_dt, w_dt, b_dt = ctx.grad_dtypes

        def _match(
            g: Optional[torch.Tensor], dt: torch.dtype
        ) -> Optional[torch.Tensor]:
            return None if g is None else g.to(dt)

        return (
            _match(dx, x_dt),
            _match(dlin_w, w_dt),
            _match(dlin_b, b_dt),
            _match(dnorm_w, norm_w.dtype),
            _match(dnorm_b, norm_b.dtype),
            None,
            None,
            # F2's residual grad. `out = LN(h) + r` means dL/dr == dL/dout, so
            # this is the incoming gradient itself -- no kernel, no saved
            # tensor. needs[7] is consulted because the residual at the call
            # site may be a tensor that does not require grad (it does here,
            # but the Function must not fabricate a gradient for one that
            # doesn't).
            _match(dy, ctx.residual_dtype) if needs[7] else None,
        )


class FusedLinearNormMLP(torch.nn.Sequential):
    """Sequential(Linear, Norm, Linear, Norm) that fuses each pair's backward.

    Drop-in for the plain nn.Sequential it replaces: same children, same order,
    same state_dict keys. Falls back to nn.Sequential.forward whenever the fused
    path is not exactly equivalent -- see _can_fuse.
    """

    def __init__(self, *modules: torch.nn.Module) -> None:
        super().__init__(*modules)
        self._f1_enabled: bool = True
        self._f1_reported: bool = False

    def _pairs(self) -> List[Tuple[torch.nn.Linear, torch.nn.Module]]:
        kids = list(self)
        return [(kids[i], kids[i + 1]) for i in range(0, len(kids), 2)]

    def _can_fuse(self, x: torch.Tensor) -> bool:
        if not self._f1_enabled or not x.is_cuda or x.dim() != 2:
            return False
        kids = list(self)
        if len(kids) % 2 != 0:
            return False
        for lin, norm in self._pairs():
            if not isinstance(lin, torch.nn.Linear) or lin.bias is None:
                return False
            if not isinstance(norm, (LayerNorm, SwishLayerNorm)):
                return False
            # The fused backward hard-codes the Triton norm kernels. If this
            # module is running on the PyTorch or AOT ladder rung, fusing would
            # silently switch kernels underneath the caller.
            if norm.hammer_kernel() != HammerKernel.TRITON:
                return False
        return True

    def _report_once(self, fused: bool, x: torch.Tensor) -> None:
        """Say, from inside the fused path, whether it actually fused.

        make_model's "armed OK: fuse_linear_norm ACTIVE" line fires at model
        CONSTRUCTION -- it proves the gin flag reached the process, and nothing
        more. _can_fuse is evaluated per forward and bails to nn.Sequential on
        the non-Triton ladder rungs, so a run can print ACTIVE and still never
        fuse a single pair. That is the inert-arm failure mode in its most
        flattering form: the campaign's arming check passes, the A/B measures
        nothing, and the delta reads as a null result for the optimization
        rather than for the plumbing. This line is emitted by the code that
        does the work, which is the only kind of proof worth having.
        """
        self._f1_reported = True
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        if fused:
            logger.info(
                "armed OK: fuse_linear_norm FUSED -- %d (Linear, Norm) pairs "
                "fused in this container, x=%s; each Linear's bias grad now "
                "comes from the norm backward's in-register column sum",
                len(self._pairs()), tuple(x.shape),
            )
        else:
            logger.warning(
                "fuse_linear_norm INERT -- _can_fuse declined (enabled=%s "
                "cuda=%s dim=%d children=%d); this container ran the shipped "
                "nn.Sequential path and measures nothing",
                self._f1_enabled, x.is_cuda, x.dim(), len(list(self)),
            )

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """F2: `mlp(x, residual=r)` computes `mlp(x) + r` in the last norm.

        The caller's `out = out + mlp(a)` becomes `out = mlp(a, residual=out)`,
        which is the same value with one fewer [M, D] pass over HBM. `residual`
        belongs to the LAST pair only -- it is added to the container's OUTPUT,
        not to each intermediate norm -- so it is threaded to the final
        iteration and nowhere else.

        When the fused path declines (`_can_fuse` false, e.g. a non-Triton
        ladder rung) the residual is added the shipped way, outside. That is a
        fallback in the mechanism, not in the semantics: the value is identical
        either way, so no arm can be inert without the F1 report saying so.
        """
        can = self._can_fuse(x)
        if not self._f1_reported:
            self._report_once(can, x)
        if not can:
            y = super().forward(x)
            return y if residual is None else y + residual
        pairs = self._pairs()
        last = len(pairs) - 1
        for i, (lin, norm) in enumerate(pairs):
            x = _LinearNormFunction.apply(
                x,
                lin.weight,
                lin.bias,
                norm.weight,
                norm.bias,
                norm._eps,
                isinstance(norm, SwishLayerNorm),
                residual if i == last else None,
            )
        return x
