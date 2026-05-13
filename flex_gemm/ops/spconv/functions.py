from typing import *
import itertools
from abc import abstractmethod

import torch
from torch import Tensor
from torch.autograd import Function
from .. import spconv
from ... import config
from ..utils import make_conv_kernel_delta, init_hashmap, lookup_pytorch
from ... import kernels
from .neighbor_cache import SparseConvNeighborCache


__all__ = [
    "sparse_submanifold_conv3d",
    "sparse_submanifold_conv",
    "sparse_submanifold_conv_any_offset",
]


class SparseConvExplicitGemmFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        neighbor_cache: SparseConvNeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SparseConvNeighborCache]:
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        neighbor_map = neighbor_cache.fwd_neighbor_map
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
        neighbor_cache: SparseConvNeighborCache = ctx.neighbor_cache
        neighbor_map = neighbor_cache.fwd_neighbor_map
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

        return grad_input, None, grad_weight, grad_bias


class SparseConvImplicitGemmFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        neighbor_cache: SparseConvNeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SparseConvNeighborCache]:
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        output  = kernels.triton.sparse_conv_fwd_implicit_gemm(
            input,
            weight,
            bias,
            neighbor_cache.fwd_neighbor_map
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: Tensor, _):
        input, weight, bias = ctx.saved_tensors
        neighbor_cache: SparseConvNeighborCache = ctx.neighbor_cache

        grad_output = grad_output.contiguous()
        if input.requires_grad:
            if neighbor_cache.symmetric:
                grad_input = kernels.triton.sparse_conv_bwd_input_implicit_gemm(
                    grad_output,
                    weight,
                    symmetric=True,
                    fwd_neighbor_map=neighbor_cache.fwd_neighbor_map,
                )
            else:
                grad_input = kernels.triton.sparse_conv_bwd_input_implicit_gemm(
                    grad_output,
                    weight,
                    symmetric=False,
                    bwd_neighbor_map=neighbor_cache.bwd_neighbor_map,
                )
        if weight.requires_grad:
            grad_weight = kernels.triton.sparse_conv_bwd_weight_implicit_gemm(
                grad_output, 
                input, 
                neighbor_cache.fwd_neighbor_map
            )
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, None, grad_weight, grad_bias


class SparseConvImplicitGemmSplitKFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: Tensor,
        neighbor_cache: SparseConvNeighborCache,
        weight: Tensor,
        bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, SparseConvNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_conv_fwd_implicit_gemm_splitk(
            feats,
            weight,
            bias,
            neighbor_cache.fwd_neighbor_map
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: Tensor, _):
        input, weight, bias = ctx.saved_tensors
        neighbor_cache: SparseConvNeighborCache = ctx.neighbor_cache

        grad_output = grad_output.contiguous()
        if input.requires_grad:
            if neighbor_cache.symmetric:
                grad_input = kernels.triton.sparse_conv_bwd_input_implicit_gemm_splitk(
                    grad_output,
                    weight,
                    symmetric=True,
                    fwd_neighbor_map=neighbor_cache.fwd_neighbor_map,
                )
            else:
                grad_input = kernels.triton.sparse_conv_bwd_input_implicit_gemm_splitk(
                    grad_output,
                    weight,
                    symmetric=False,
                    bwd_neighbor_map=neighbor_cache.bwd_neighbor_map,
                )
        if weight.requires_grad:
            grad_weight = kernels.triton.sparse_conv_bwd_weight_implicit_gemm_splitk(
                grad_output,
                input,
                neighbor_cache.fwd_neighbor_map
            )
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, None, grad_weight, grad_bias


class SparseConvMaskedImplicitGemmFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        neighbor_cache: SparseConvNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SparseConvNeighborCache]:
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_conv_fwd_masked_implicit_gemm(
            input,
            weight,
            bias,
            neighbor_cache.fwd_neighbor_map,
            neighbor_cache.fwd_sorted_idx,
            neighbor_cache.fwd_valid_kernel_callback,
            neighbor_cache.fwd_valid_kernel_seg_callback,
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        input, weight, bias = ctx.saved_tensors
        neighbor_cache: SparseConvNeighborCache = ctx.neighbor_cache

        grad_output = grad_output.contiguous()
        if input.requires_grad:
            if neighbor_cache.symmetric:
                grad_input = kernels.triton.sparse_conv_bwd_input_masked_implicit_gemm(
                    grad_output,
                    weight,
                    symmetric=True,
                    fwd_neighbor_map=neighbor_cache.fwd_neighbor_map,
                    fwd_sorted_idx=neighbor_cache.fwd_sorted_idx,
                    fwd_valid_kernel=neighbor_cache.fwd_valid_kernel_callback,
                    fwd_valid_kernel_seg=neighbor_cache.fwd_valid_kernel_seg_callback,
                )
            else:
                grad_input = kernels.triton.sparse_conv_bwd_input_masked_implicit_gemm(
                    grad_output,
                    weight,
                    symmetric=False,
                    bwd_neighbor_map=neighbor_cache.bwd_neighbor_map,
                    bwd_sorted_idx=neighbor_cache.bwd_sorted_idx,
                    bwd_valid_kernel=neighbor_cache.bwd_valid_kernel_callback,
                    bwd_valid_kernel_seg=neighbor_cache.bwd_valid_kernel_seg_callback,
                )
        if weight.requires_grad:
            grad_weight = kernels.triton.sparse_conv_bwd_weight_masked_implicit_gemm(
                grad_output,
                input,
                neighbor_cache.fwd_valid_signal_i,
                neighbor_cache.fwd_valid_signal_o,
                neighbor_cache.fwd_valid_signal_seg,
            )
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, None, grad_weight, grad_bias


class SparseConvMaskedImplicitGemmSplitKFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        neighbor_cache: SparseConvNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SparseConvNeighborCache]:
        assert input.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert input.shape[-1] == Ci, f"Input channels ({input.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_conv_fwd_masked_implicit_gemm_splitk(
            input,
            weight,
            bias,
            neighbor_cache.fwd_neighbor_map,
            neighbor_cache.fwd_sorted_idx,
            neighbor_cache.fwd_valid_kernel_callback,
            neighbor_cache.fwd_valid_kernel_seg_callback,
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        input, weight, bias = ctx.saved_tensors
        neighbor_cache: SparseConvNeighborCache = ctx.neighbor_cache

        grad_output = grad_output.contiguous()
        if input.requires_grad:
            if neighbor_cache.symmetric:
                grad_input = kernels.triton.sparse_conv_bwd_input_masked_implicit_gemm_splitk(
                    grad_output,
                    weight,
                    symmetric=True,
                    fwd_neighbor_map=neighbor_cache.fwd_neighbor_map,
                    fwd_sorted_idx=neighbor_cache.fwd_sorted_idx,
                    fwd_valid_kernel=neighbor_cache.fwd_valid_kernel_callback,
                    fwd_valid_kernel_seg=neighbor_cache.fwd_valid_kernel_seg_callback,
                )
            else:
                grad_input = kernels.triton.sparse_conv_bwd_input_masked_implicit_gemm_splitk(
                    grad_output,
                    weight,
                    symmetric=False,
                    bwd_neighbor_map=neighbor_cache.bwd_neighbor_map,
                    bwd_sorted_idx=neighbor_cache.bwd_sorted_idx,
                    bwd_valid_kernel=neighbor_cache.bwd_valid_kernel_callback,
                    bwd_valid_kernel_seg=neighbor_cache.bwd_valid_kernel_seg_callback,
                )
        if weight.requires_grad:
            grad_weight = kernels.triton.sparse_conv_bwd_weight_masked_implicit_gemm_splitk(
                grad_output,
                input,
                neighbor_cache.fwd_valid_signal_i,
                neighbor_cache.fwd_valid_signal_o,
                neighbor_cache.fwd_valid_signal_seg,
            )
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, None, grad_weight, grad_bias


def _select_function(algorithm: Literal["explicit_gemm", "implicit_gemm", "implicit_gemm_splitk", "masked_implicit_gemm", "masked_implicit_gemm_splitk"] | None = None) -> Type[Function]:
    if algorithm is None:
        # Default to the global config algorithm if not specified.
        algorithm = spconv.ALGORITHM
        
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

