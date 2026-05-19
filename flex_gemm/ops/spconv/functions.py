from typing import *
import itertools

import torch
from torch import Tensor
from torch.autograd import Function
from ... import config
from ... import kernels
from ..neighbor_cache import NeighborCache


class SparseConvExplicitGemmFunction(Function):
    @staticmethod
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
    def forward(
        ctx,
        input: Tensor,
        neighbor_cache: NeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[Tensor, NeighborCache]:
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
    def forward(
        ctx,
        feats: Tensor,
        neighbor_cache: NeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[Tensor, NeighborCache]:
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
    def forward(
        ctx,
        input: torch.Tensor,
        neighbor_cache: NeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, NeighborCache]:
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
    def forward(
        ctx,
        input: torch.Tensor,
        neighbor_cache: NeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        allow_tf32: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, NeighborCache]:
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

