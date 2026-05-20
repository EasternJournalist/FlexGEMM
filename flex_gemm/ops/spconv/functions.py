from typing import *
import itertools

import torch
from torch import Tensor
from torch.autograd import Function
try:
    # PyTorch >= 2.4 unified API.
    from torch.amp import custom_fwd as _custom_fwd, custom_bwd as _custom_bwd
    def custom_fwd(fn):
        return _custom_fwd(device_type="cuda")(fn)
    def custom_bwd(fn):
        return _custom_bwd(device_type="cuda")(fn)
except ImportError:  # pragma: no cover - legacy fallback
    from torch.cuda.amp import custom_fwd, custom_bwd  # type: ignore[no-redef]
from ... import config
from ... import kernels
from ..neighbor_cache import NeighborCache


def _amp_cast(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    """Resolve the compute dtype under ``torch.autocast`` and cast inputs.

    ``custom_fwd`` attaches ``ctx._fwd_used_autocast`` / ``ctx._dtype`` that
    record whether the *caller* was inside an autocast region and which dtype
    was active. Inside the decorated ``forward`` autocast is already disabled,
    so we must rely on those captured attributes (not
    ``torch.is_autocast_enabled()``).

    Returns ``(input, weight, bias)`` cast to a common compute dtype:
      * autocast active  -> ``ctx._dtype`` (fp16 or bf16, set by the user)
      * autocast off     -> ``input.dtype`` (no-op, just enforces weight/bias)

    ``bias`` is **intentionally not cast** — the Triton fwd kernels add bias
    to the fp32 accumulator in the GEMM epilogue, so keeping bias in its
    original (typically fp32) dtype preserves precision. Passing fp32 bias
    when input/weight are fp16 only costs a negligible HBM read per output
    tile and is the same epilogue contract used by cuBLAS Lt and PyTorch's
    ``nn.Linear`` under autocast.

    Crucially, ``requires_grad`` from the *original* forward arguments is
    propagated onto the cast tensors. Inside ``Function.forward`` autograd is
    disabled, so ``.to(other_dtype)`` returns a leaf-like tensor with
    ``requires_grad=False``; without re-attaching the flag, the existing
    ``if X.requires_grad:`` branches in ``backward`` (which inspect saved
    tensors) would all become False under AMP, silently producing ``None``
    grads.
    """
    if getattr(ctx, "_fwd_used_autocast", False):
        compute_dtype = ctx._dtype
    else:
        compute_dtype = input.dtype

    inp_rg = input.requires_grad
    w_rg = weight.requires_grad
    b_rg = bias is not None and bias.requires_grad

    input = input.to(compute_dtype)
    weight = weight.to(compute_dtype)
    # bias intentionally kept at its original dtype (see docstring).

    if inp_rg:
        input.requires_grad_(True)
    if w_rg:
        weight.requires_grad_(True)
    if bias is not None and b_rg:
        bias.requires_grad_(True)

    return input, weight, bias


class SparseConvExplicitGemmFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: Tensor,
        neighbor_cache: NeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[Tensor, NeighborCache]:
        # ``allow_tf32`` is accepted for API uniformity but has no effect on
        # this path: the explicit-GEMM variant defers matmuls to
        # :func:`torch.mm` / :func:`torch.addmm`, which respect the global
        # ``torch.backends.cuda.matmul.allow_tf32`` switch rather than our
        # SPCONV_ALLOW_TF32 config.
        del allow_tf32
        input, weight, bias = _amp_cast(ctx, input, weight, bias)
        input = input.contiguous()
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        neighbor_map = neighbor_cache.fwd_map
        N = input.shape[0]
        im2col = input.index_select(0, neighbor_map.view(-1).view(dtype=torch.int32).clamp_min(0))\
                        .masked_fill((neighbor_map == -1).view(-1, 1), 0).view(N, V * Ci)

        weight_mat = weight.view(Co, V * Ci).transpose(0, 1)
        if bias is not None:
            output = torch.addmm(bias, im2col, weight_mat)
        else:
            output = torch.mm(im2col, weight_mat)

        ctx.save_for_backward(input, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor, _):
        input, weight, bias = ctx.saved_tensors
        neighbor_cache: NeighborCache = ctx.neighbor_cache
        neighbor_map = neighbor_cache.fwd_map
        N = input.shape[0]
        Co, V, Ci = weight.shape

        if input.requires_grad:
            im2col = torch.zeros((N * V, Co), device=input.device, dtype=input.dtype)
            inv_neighbor_map = torch.flip(neighbor_map, [1])
            mask = inv_neighbor_map.view(-1) != -1
            im2col[mask] = grad_output[inv_neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(N, V * Co)
            grad_input = torch.mm(im2col, weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci))
        else:
            grad_input = None

        if weight.requires_grad:
            im2col = torch.zeros((N * V, Ci), device=weight.device, dtype=weight.dtype)
            mask = neighbor_map.view(-1) != -1
            im2col[mask] = input[neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(N, V * Ci)
            grad_weight = torch.mm(im2col.t(), grad_output.view(N, -1)).view(V, Ci, Co).permute(2, 0, 1).contiguous()
        else:
            grad_weight = None

        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None


class SparseConvImplicitGemmFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: Tensor,
        neighbor_cache: NeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[Tensor, NeighborCache]:
        input, weight, bias = _amp_cast(ctx, input, weight, bias)
        input = input.contiguous()
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        output  = kernels.triton.sparse_conv_fwd_implicit_gemm(
            input,
            weight,
            bias,
            neighbor_cache.fwd_map,
            allow_tf32=allow_tf32,
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        ctx.allow_tf32 = allow_tf32
        return output, neighbor_cache

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor, _):
        input, weight, bias = ctx.saved_tensors
        neighbor_cache: NeighborCache = ctx.neighbor_cache
        allow_tf32 = ctx.allow_tf32

        grad_output = grad_output.contiguous()
        
        if input.requires_grad:
            if neighbor_cache.symmetric:
                grad_input = kernels.triton.sparse_conv_bwd_input_implicit_gemm(
                    grad_output,
                    weight,
                    symmetric=True,
                    fwd_neighbor_map=neighbor_cache.fwd_map,
                    allow_tf32=allow_tf32,
                )
            else:
                grad_input = kernels.triton.sparse_conv_bwd_input_implicit_gemm(
                    grad_output,
                    weight,
                    symmetric=False,
                    bwd_neighbor_map=neighbor_cache.bwd_map,
                    allow_tf32=allow_tf32,
                )
        else:
            grad_input = None

        if weight.requires_grad:
            grad_weight = kernels.triton.sparse_conv_bwd_weight_implicit_gemm(
                grad_output, 
                input, 
                neighbor_cache.fwd_map,
                allow_tf32=allow_tf32,
            )
        else:
            grad_weight = None

        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None


class SparseConvImplicitGemmSplitKFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        feats: Tensor,
        neighbor_cache: NeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[Tensor, NeighborCache]:
        feats, weight, bias = _amp_cast(ctx, feats, weight, bias)
        feats = feats.contiguous()
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_conv_fwd_implicit_gemm_splitk(
            feats,
            weight,
            bias,
            neighbor_cache.fwd_map,
            allow_tf32=allow_tf32,
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        ctx.allow_tf32 = allow_tf32
        return output, neighbor_cache

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: Tensor, _):
        input, weight, bias = ctx.saved_tensors
        neighbor_cache: NeighborCache = ctx.neighbor_cache
        allow_tf32 = ctx.allow_tf32

        grad_output = grad_output.contiguous()
        if input.requires_grad:
            if neighbor_cache.symmetric:
                grad_input = kernels.triton.sparse_conv_bwd_input_implicit_gemm_splitk(
                    grad_output,
                    weight,
                    symmetric=True,
                    fwd_neighbor_map=neighbor_cache.fwd_map,
                    allow_tf32=allow_tf32,
                )
            else:
                grad_input = kernels.triton.sparse_conv_bwd_input_implicit_gemm_splitk(
                    grad_output,
                    weight,
                    symmetric=False,
                    bwd_neighbor_map=neighbor_cache.bwd_map,
                    allow_tf32=allow_tf32,
                )
        else:
            grad_input = None

        if weight.requires_grad:
            grad_weight = kernels.triton.sparse_conv_bwd_weight_implicit_gemm_splitk(
                grad_output,
                input,
                neighbor_cache.fwd_map,
                allow_tf32=allow_tf32,
            )
        else:
            grad_weight = None

        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None


class SparseConvMaskedImplicitGemmFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        neighbor_cache: NeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, NeighborCache]:
        input, weight, bias = _amp_cast(ctx, input, weight, bias)
        input = input.contiguous()
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_conv_fwd_masked_implicit_gemm(
            input,
            weight,
            bias,
            neighbor_cache.fwd_map,
            neighbor_cache.fwd_sorted_idx,
            neighbor_cache.fwd_valid_kernel_callback,
            neighbor_cache.fwd_valid_kernel_seg_callback,
            allow_tf32=allow_tf32,
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        ctx.allow_tf32 = allow_tf32
        return output, neighbor_cache

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor, _):
        input, weight, bias = ctx.saved_tensors
        neighbor_cache: NeighborCache = ctx.neighbor_cache
        allow_tf32 = ctx.allow_tf32

        grad_output = grad_output.contiguous()
        if input.requires_grad:
            if neighbor_cache.symmetric:
                grad_input = kernels.triton.sparse_conv_bwd_input_masked_implicit_gemm(
                    grad_output,
                    weight,
                    symmetric=True,
                    fwd_neighbor_map=neighbor_cache.fwd_map,
                    fwd_sorted_idx=neighbor_cache.fwd_sorted_idx,
                    fwd_valid_kernel=neighbor_cache.fwd_valid_kernel_callback,
                    fwd_valid_kernel_seg=neighbor_cache.fwd_valid_kernel_seg_callback,
                    allow_tf32=allow_tf32,
                )
            else:
                grad_input = kernels.triton.sparse_conv_bwd_input_masked_implicit_gemm(
                    grad_output,
                    weight,
                    symmetric=False,
                    bwd_neighbor_map=neighbor_cache.bwd_map,
                    bwd_sorted_idx=neighbor_cache.bwd_sorted_idx,
                    bwd_valid_kernel=neighbor_cache.bwd_valid_kernel_callback,
                    bwd_valid_kernel_seg=neighbor_cache.bwd_valid_kernel_seg_callback,
                    allow_tf32=allow_tf32,
                )
        else:
            grad_input = None
                
        if weight.requires_grad:
            grad_weight = kernels.triton.sparse_conv_bwd_weight_masked_implicit_gemm(
                grad_output,
                input,
                neighbor_cache.fwd_valid_signal_i,
                neighbor_cache.fwd_valid_signal_o,
                neighbor_cache.fwd_valid_signal_seg,
                allow_tf32=allow_tf32,
            )
        else:
            grad_weight = None

        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias, None


class SparseConvMaskedImplicitGemmSplitKFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        neighbor_cache: NeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, NeighborCache]:
        input, weight, bias = _amp_cast(ctx, input, weight, bias)
        input = input.contiguous()
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_conv_fwd_masked_implicit_gemm_splitk(
            input,
            weight,
            bias,
            neighbor_cache.fwd_map,
            neighbor_cache.fwd_sorted_idx,
            neighbor_cache.fwd_valid_kernel_callback,
            neighbor_cache.fwd_valid_kernel_seg_callback,
            allow_tf32=allow_tf32,
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        ctx.allow_tf32 = allow_tf32
        return output, neighbor_cache

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor, _):
        input, weight, bias = ctx.saved_tensors
        neighbor_cache: NeighborCache = ctx.neighbor_cache
        allow_tf32 = ctx.allow_tf32

        grad_output = grad_output.contiguous()
        if input.requires_grad:
            if neighbor_cache.symmetric:
                grad_input = kernels.triton.sparse_conv_bwd_input_masked_implicit_gemm_splitk(
                    grad_output,
                    weight,
                    symmetric=True,
                    fwd_neighbor_map=neighbor_cache.fwd_map,
                    fwd_sorted_idx=neighbor_cache.fwd_sorted_idx,
                    fwd_valid_kernel=neighbor_cache.fwd_valid_kernel_callback,
                    fwd_valid_kernel_seg=neighbor_cache.fwd_valid_kernel_seg_callback,
                    allow_tf32=allow_tf32,
                )
            else:
                grad_input = kernels.triton.sparse_conv_bwd_input_masked_implicit_gemm_splitk(
                    grad_output,
                    weight,
                    symmetric=False,
                    bwd_neighbor_map=neighbor_cache.bwd_map,
                    bwd_sorted_idx=neighbor_cache.bwd_sorted_idx,
                    bwd_valid_kernel=neighbor_cache.bwd_valid_kernel_callback,
                    bwd_valid_kernel_seg=neighbor_cache.bwd_valid_kernel_seg_callback,
                    allow_tf32=allow_tf32,
                )
        else:
            grad_input = None

        if weight.requires_grad:
            grad_weight = kernels.triton.sparse_conv_bwd_weight_masked_implicit_gemm_splitk(
                grad_output,
                input,
                neighbor_cache.fwd_valid_signal_i,
                neighbor_cache.fwd_valid_signal_o,
                neighbor_cache.fwd_valid_signal_seg,
                allow_tf32=allow_tf32,
            )
        else:
            grad_weight = None

        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None
            
        return grad_input, None, grad_weight, grad_bias, None


def _select_function(algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] | None = None) -> Type[Function]:
    if algorithm is None:
        # Default to the global config algorithm if not specified.
        algorithm = config.DEFAULT_SPCONV_ALGORITHM
        
    if algorithm == "explicit_gemm":
        return SparseConvExplicitGemmFunction
    if algorithm == "implicit_gemm":
        return SparseConvImplicitGemmFunction
    if algorithm == "implicit_gemm_splitk":
        return SparseConvImplicitGemmSplitKFunction
    if algorithm == "masked_implicit_gemm":
        return SparseConvMaskedImplicitGemmFunction
    if algorithm == "masked_implicit_gemm_splitk":
        return SparseConvMaskedImplicitGemmSplitKFunction
    raise ValueError(f"Invalid algorithm {algorithm}")

