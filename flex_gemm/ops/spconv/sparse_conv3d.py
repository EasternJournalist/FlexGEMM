from typing import *
import torch
from torch.autograd import Function
from . import Algorithm, SparseConv3dOutCoordAlgorithm
from .. import spconv
from ... import kernels


__all__ = [
    "SparseConv3dNeighborCache",
    "SparseConv3dExplicitGemmFunction",
    "SparseConv3dImplicitGemmFunction",
    "SparseConv3dImplicitGemmSplitKFunction",
    "SparseConv3dMaskedImplicitGemmFunction",
    "SparseConv3dMaskedImplicitGemmSplitKFunction",
    "sparse_conv3d",
    "SparseConv3dFunction",
]


class SparseConv3dNeighborCache:
    neighbor_map: torch.Tensor
    neighbor_map_bwd: Optional[torch.Tensor]

    def __init__(self, neighbor_map: torch.Tensor, neighbor_map_bwd: Optional[torch.Tensor] = None, needs_bwd_sort: bool = True):
        self.neighbor_map = neighbor_map
        self.neighbor_map_bwd = neighbor_map_bwd
        self._needs_bwd_sort = needs_bwd_sort

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def neighbor_map_post_process_for_masked_implicit_gemm_1(self):
        """Compute and cache the forward masked-IGEMM auxiliary tensors (idempotent)."""
        if 'gray_code' in self:
            return
        neighbor_map = self.neighbor_map
        V = neighbor_map.shape[1]
        assert V <= 32, "Currently, the max kernel volume is 32 because kernel mask is encoded as uint32"
        gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg = \
            kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1(neighbor_map)
        self['gray_code'] = gray_code
        self['sorted_idx'] = sorted_idx
        self['valid_signal_i'] = valid_signal_i
        self['valid_signal_o'] = valid_signal_o
        self['valid_signal_seg'] = valid_signal_seg

    def neighbor_map_post_process_for_masked_implicit_gemm_1_bwd(self):
        """Compute and cache the backward masked-IGEMM auxiliary tensors (idempotent).

        Only runs if ``needs_bwd_sort`` is True.  When stride == 1 and output
        coords are auto-generated, backward masks are nearly all valid and
        sorting is unnecessary.
        """
        if not self._needs_bwd_sort or 'gray_code_bwd' in self:
            return
        gray_code_bwd, sorted_idx_bwd = \
            kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd(self.neighbor_map_bwd)
        self['gray_code_bwd'] = gray_code_bwd
        self['sorted_idx_bwd'] = sorted_idx_bwd

    # NOTE:
    # valid_kernel and valid_kernel_seg are block-size dependent because
    # Triton kernels use different block-sizes during autotuning.
    #
    # We lazily compute and cache them here to:
    #   1. Avoid recomputation across multiple kernel launches
    #   2. Support multiple Triton specializations with the same neighbor cache

    def compute_kernel_idx(self, block_size: int):
        valid_kernel, valid_kernel_seg = kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2(
            self['gray_code'], self['sorted_idx'], block_size)
        self[f'valid_kernel_{block_size}'] = valid_kernel
        self[f'valid_kernel_seg_{block_size}'] = valid_kernel_seg

    def valid_kernel_callback(self, block_size: int) -> torch.Tensor:
        if f'valid_kernel_{block_size}' not in self:
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_{block_size}']

    def valid_kernel_seg_callback(self, block_size: int) -> torch.Tensor:
        if f'valid_kernel_{block_size}' not in self:
            self.compute_kernel_idx(block_size)
        return self[f'valid_kernel_seg_{block_size}']

    def compute_kernel_idx_bwd(self, block_size: int):
        valid_kernel, valid_kernel_seg = kernels.cuda.neighbor_map_post_process_for_masked_implicit_gemm_2(
            self['gray_code_bwd'], self['sorted_idx_bwd'], block_size)
        self[f'valid_kernel_bwd_{block_size}'] = valid_kernel
        self[f'valid_kernel_bwd_seg_{block_size}'] = valid_kernel_seg

    def valid_kernel_bwd_callback(self, block_size: int) -> torch.Tensor:
        if f'valid_kernel_bwd_{block_size}' not in self:
            self.compute_kernel_idx_bwd(block_size)
        return self[f'valid_kernel_bwd_{block_size}']

    def valid_kernel_bwd_seg_callback(self, block_size: int) -> torch.Tensor:
        if f'valid_kernel_bwd_{block_size}' not in self:
            self.compute_kernel_idx_bwd(block_size)
        return self[f'valid_kernel_bwd_seg_{block_size}']


def _get_output_coords(
    coords: torch.Tensor,
    shape: torch.Size,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> torch.Tensor:
    assert coords.is_contiguous(), "Coords should be contiguous"
    assert coords.dtype in [torch.int32], "Unsupported coords dtype. Expect int32"
    N, C, W, H, D = shape

    if coords.is_cuda:
        if spconv.OUT_COORD_ALGO == SparseConv3dOutCoordAlgorithm.HASHMAP:
            output_coords = kernels.cuda.hashmap_build_sparse_conv_out_coords(
                coords, spconv.OUT_COORD_HASHMAP_RATIO, spconv.SERIALIZATION_MODE,
                N, W, H, D,
                kernel_size[0], kernel_size[1], kernel_size[2],
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
        elif spconv.OUT_COORD_ALGO == SparseConv3dOutCoordAlgorithm.EXPAND_UNIQUE:
            output_coords = kernels.cuda.expand_unique_build_sparse_conv_out_coords(
                coords, spconv.SERIALIZATION_MODE,
                N, W, H, D,
                kernel_size[0], kernel_size[1], kernel_size[2],
                stride[0], stride[1], stride[2],
                padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2],
            )
    else:
        raise NotImplementedError("CPU version is not implemented")
    return output_coords


def _get_output_coords_torch(
    coords: torch.Tensor,
    shape: torch.Size,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> torch.Tensor:
    N, C, W, H, D = shape
    Wo = (W + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    Ho = (H + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    Do = (D + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2] + 1

    delta = torch.meshgrid(
        -dilation[0] * torch.arange(kernel_size[0]),
        -dilation[1] * torch.arange(kernel_size[1]),
        -dilation[2] * torch.arange(kernel_size[2]),
        indexing='ij'
    )
    delta = torch.stack(delta, dim=-1).reshape(-1, 3).int().to(coords.device)
    all_potentials_out_coords = (coords + torch.tensor([0, padding[0], padding[1], padding[2]], device=coords.device).int()) \
                                 .unsqueeze(1).repeat(1, kernel_size[0] * kernel_size[1] * kernel_size[2], 1)
    all_potentials_out_coords[:, :, 1:] += delta.unsqueeze(0)                          # [N, kernel_vol, 4]
    all_potentials_out_coords = all_potentials_out_coords.reshape(-1, 4)                # [N * kernel_vol, 4]
    t_stride = torch.tensor([1, stride[0], stride[1], stride[2]], device=coords.device).int()
    valid_strided_out_coords = torch.all(all_potentials_out_coords % t_stride == 0, dim=-1)
    all_potentials_out_coords = all_potentials_out_coords[valid_strided_out_coords] // t_stride
    t_out_size = torch.tensor([Wo, Ho, Do], device=coords.device).int()
    valid_out_coords = torch.all((all_potentials_out_coords[:, 1:] >= 0) * (all_potentials_out_coords[:, 1:] < t_out_size), dim=-1)
    all_potentials_out_coords = all_potentials_out_coords[valid_out_coords]
    out_coords = torch.unique(all_potentials_out_coords, dim=0)

    return out_coords


def _compute_neighbor_cache(
    coords: torch.Tensor,
    out_coords: torch.Tensor,
    shape: torch.Size,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    needs_grad: bool,
    needs_bwd_sort: bool = False,
) -> SparseConv3dNeighborCache:
    assert coords.is_contiguous(), "Coords should be contiguous"
    assert coords.dtype in [torch.int32], "Unsupported coords dtype. Expect int32"
    N, C, W, H, D = shape

    if coords.is_cuda:
        neighbor_map, neighbor_map_bwd = kernels.cuda.hashmap_build_sparse_conv_neighbour_map(
            coords, out_coords, spconv.HASHMAP_RATIO, needs_grad,
            N, W, H, D,
            kernel_size[0], kernel_size[1], kernel_size[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
        )
    else:
        raise NotImplementedError("CPU version of hashmap is not implemented")

    return SparseConv3dNeighborCache(neighbor_map, neighbor_map_bwd, needs_bwd_sort=needs_bwd_sort)


def _compute_neighbor_cache_torch(
    coords: torch.Tensor,
    out_coords: torch.Tensor,
    shape: torch.Size,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    needs_grad: bool,
) -> SparseConv3dNeighborCache:
    N, C, W, H, D = shape
    M = coords.shape[0]
    L = out_coords.shape[0]
    V = kernel_size[0] * kernel_size[1] * kernel_size[2]
    assert N * W * H * D <= 2**32, "Currently, the max number of elements in a tensor is 2^32"
    OFFSET = torch.tensor([W * H * D, H * D, D, 1], device=coords.device).int()

    keys = (coords * OFFSET).sum(dim=-1)
    sorted_keys, indices = torch.sort(keys)

    # Compute neighbor coords
    offset = torch.meshgrid(
        torch.arange(kernel_size[0]) * dilation[0] - padding[0],
        torch.arange(kernel_size[1]) * dilation[1] - padding[1],
        torch.arange(kernel_size[2]) * dilation[2] - padding[2],
        indexing='ij'
    )
    offset = torch.stack(offset, dim=-1).reshape(-1, 3).int().to(coords.device)
    t_stride = torch.tensor([1, stride[0], stride[1], stride[2]], device=coords.device).int()
    neighbor_coords = (out_coords * t_stride).unsqueeze(1).repeat(1, V, 1)
    neighbor_coords[:, :, -3:] += offset.unsqueeze(0)                                    # [L, V, 4]
    neighbor_coords = neighbor_coords.reshape(-1, 4)                                     # [L * V, 4]
    neighbor_valid = (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < W) & \
                     (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < H) & \
                     (neighbor_coords[:, 3] >= 0) & (neighbor_coords[:, 3] < D)
    neighbor_keys = (neighbor_coords * OFFSET).sum(dim=-1)
    neighbor_search_indices = torch.searchsorted(sorted_keys, neighbor_keys)
    neighbor_search_indices = torch.clamp(neighbor_search_indices, 0, sorted_keys.shape[0] - 1)
    neighbor_valid &= sorted_keys[neighbor_search_indices] == neighbor_keys
    neighbor_map = torch.full((L * V,), 0xffffffff, dtype=torch.long, device=coords.device)
    in_indices = indices[neighbor_search_indices[neighbor_valid]]
    v = torch.arange(V, device=coords.device).reshape(1, -1).repeat(L, 1).flatten()[neighbor_valid]
    out_indices = torch.arange(L, device=coords.device).reshape(-1, 1).repeat(1, V).flatten()[neighbor_valid]
    neighbor_map[out_indices * V + v] = in_indices
    if needs_grad:
        neighbor_map_bwd = torch.full((M * V,), 0xffffffff, dtype=torch.long, device=coords.device)
        neighbor_map_bwd[in_indices * V + v] = out_indices
    else:
        neighbor_map_bwd = None
    return SparseConv3dNeighborCache(
        neighbor_map=neighbor_map.reshape(L, -1).to(torch.uint32),
        neighbor_map_bwd=neighbor_map_bwd.reshape(M, -1).to(torch.uint32) if needs_grad else None,
    )


class SparseConv3dExplicitGemmFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        neighbor_cache: SparseConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SparseConv3dNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        neighbor_map = neighbor_cache['neighbor_map']

        # im2col
        im2col = torch.zeros((neighbor_map.shape[0] * V, Ci), device=feats.device, dtype=feats.dtype)
        mask = neighbor_map.view(-1) != 0xffffffff
        im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
        im2col = im2col.view(neighbor_map.shape[0], V * Ci)

        # addmm
        weight_mat = weight.view(Co, V * Ci).transpose(0, 1)
        if bias is not None:
            output = torch.addmm(bias, im2col, weight_mat)
        else:
            output = torch.mm(im2col, weight_mat)

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache
        Co, V, Ci = weight.shape
        neighbor_map = neighbor_cache['neighbor_map']
        neighbor_map_bwd = neighbor_cache['neighbor_map_bwd']

        if feats.requires_grad:
            im2col = torch.zeros((neighbor_map_bwd.shape[0] * V, Co), device=feats.device, dtype=feats.dtype)
            mask = neighbor_map_bwd.view(-1) != 0xffffffff
            im2col[mask] = grad_output[neighbor_map_bwd.view(-1).long()[mask]]
            im2col = im2col.view(neighbor_map_bwd.shape[0], V * Co)
            grad_input = torch.mm(im2col, weight.view(Co, V, Ci).transpose(0, 1).reshape(V * Co, Ci))
        else:
            grad_input = None

        if weight.requires_grad:
            im2col = torch.zeros((neighbor_map.shape[0] * V, Ci), device=weight.device, dtype=weight.dtype)
            mask = neighbor_map.view(-1) != 0xffffffff
            im2col[mask] = feats[neighbor_map.view(-1).long()[mask]]
            im2col = im2col.view(neighbor_map.shape[0], V * Ci)
            grad_weight = torch.mm(im2col.t(), grad_output.view(neighbor_map.shape[0], -1)).view(V, Ci, Co).permute(2, 0, 1).contiguous()
        else:
            grad_weight = None

        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
        else:
            grad_bias = None

        return grad_input, None, grad_weight, grad_bias


class SparseConv3dImplicitGemmFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        neighbor_cache: SparseConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SparseConv3dNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_conv_fwd_implicit_gemm(
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache

        grad_input, grad_weight, grad_bias = kernels.triton.sparse_conv_bwd_implicit_gemm(
            grad_output.contiguous(),
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['neighbor_map_bwd'],
        )

        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, grad_weight, grad_bias


class SparseConv3dImplicitGemmSplitKFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        neighbor_cache: SparseConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SparseConv3dNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        output = kernels.triton.sparse_conv_fwd_implicit_gemm_splitk(
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache

        grad_input, grad_weight, grad_bias = kernels.triton.sparse_conv_bwd_implicit_gemm_splitk(
            grad_output.contiguous(),
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['neighbor_map_bwd'],
        )

        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, grad_weight, grad_bias


class SparseConv3dMaskedImplicitGemmFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        neighbor_cache: SparseConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SparseConv3dNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        neighbor_cache.neighbor_map_post_process_for_masked_implicit_gemm_1()

        output = kernels.triton.sparse_conv_fwd_masked_implicit_gemm(
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['sorted_idx'],
            neighbor_cache.valid_kernel_callback,
            neighbor_cache.valid_kernel_seg_callback,
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache

        neighbor_cache.neighbor_map_post_process_for_masked_implicit_gemm_1_bwd()

        grad_input, grad_weight, grad_bias = kernels.triton.sparse_conv_bwd_masked_implicit_gemm(
            grad_output.contiguous(),
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['neighbor_map_bwd'],
            neighbor_cache['valid_signal_i'],
            neighbor_cache['valid_signal_o'],
            neighbor_cache['valid_signal_seg'],
            neighbor_cache['sorted_idx_bwd'],
            neighbor_cache.valid_kernel_bwd_callback,
            neighbor_cache.valid_kernel_bwd_seg_callback,
        )

        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, grad_weight, grad_bias


class SparseConv3dMaskedImplicitGemmSplitKFunction(Function):
    @staticmethod
    def forward(
        ctx,
        feats: torch.Tensor,
        neighbor_cache: SparseConv3dNeighborCache,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SparseConv3dNeighborCache]:
        assert feats.is_contiguous(), "Input features should be contiguous"
        Co, V, Ci = weight.shape
        assert feats.shape[-1] == Ci, f"Input channels ({feats.shape[-1]}) should match weight channels ({Ci})"

        neighbor_cache.neighbor_map_post_process_for_masked_implicit_gemm_1()

        output = kernels.triton.sparse_conv_fwd_masked_implicit_gemm_splitk(
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['sorted_idx'],
            neighbor_cache.valid_kernel_callback,
            neighbor_cache.valid_kernel_seg_callback,
        )

        ctx.save_for_backward(feats, weight, bias)
        ctx.neighbor_cache = neighbor_cache
        return output, neighbor_cache

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache

        neighbor_cache.neighbor_map_post_process_for_masked_implicit_gemm_1_bwd()

        grad_input, grad_weight, grad_bias = kernels.triton.sparse_conv_bwd_masked_implicit_gemm_splitk(
            grad_output.contiguous(),
            feats,
            weight,
            bias,
            neighbor_cache['neighbor_map'],
            neighbor_cache['neighbor_map_bwd'],
            neighbor_cache['valid_signal_i'],
            neighbor_cache['valid_signal_o'],
            neighbor_cache['valid_signal_seg'],
            neighbor_cache['sorted_idx_bwd'],
            neighbor_cache.valid_kernel_bwd_callback,
            neighbor_cache.valid_kernel_bwd_seg_callback,
        )

        if not feats.requires_grad:
            grad_input = None
        if not weight.requires_grad:
            grad_weight = None
        if not bias.requires_grad:
            grad_bias = None
        return grad_input, None, grad_weight, grad_bias


def _select_spconv_function(algorithm: Optional[str] = None) -> Type[Function]:
    if algorithm is None:
        algorithm = spconv.ALGORITHM

    if algorithm == Algorithm.EXPLICIT_GEMM:
        return SparseConv3dExplicitGemmFunction
    if algorithm == Algorithm.IMPLICIT_GEMM:
        return SparseConv3dImplicitGemmFunction
    if algorithm == Algorithm.IMPLICIT_GEMM_SPLITK:
        return SparseConv3dImplicitGemmSplitKFunction
    if algorithm == Algorithm.MASKED_IMPLICIT_GEMM:
        return SparseConv3dMaskedImplicitGemmFunction
    if algorithm == Algorithm.MASKED_IMPLICIT_GEMM_SPLITK:
        return SparseConv3dMaskedImplicitGemmSplitKFunction
    raise ValueError(f"Invalid algorithm {algorithm}")


def sparse_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: torch.Size,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    neighbor_cache: Optional[SparseConv3dNeighborCache] = None,
    out_coords: Optional[torch.Tensor] = None,
    stride: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int] = (0, 0, 0),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    algorithm: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, SparseConv3dNeighborCache]:
    """
    Sparse convolution for 3D input.

    Args:
        feats (torch.Tensor): [N, C] tensor of input features.
        coords (torch.Tensor): [N, 4] tensor of input coordinates.
        shape (torch.Size): shape of the input tensor in NCWHD order.
        weight (torch.Tensor): [Co, Kw, Kh, Kd, Ci] tensor of weights.
        bias (Optional[torch.Tensor]): [Co] tensor of biases.
        neighbor_cache (Optional[SparseConv3dNeighborCache]): neighbor cache for this operation.
            Can be reused for multiple runs using the same coordinates.
            If None, will be computed on the fly.
        out_coords (Optional[torch.Tensor]): [M, 4] tensor of output coordinates.
            If None, will be calculated based on the input shape, kernel size, stride, padding, and dilation.
            If specified, will be used as the output coordinates.
        stride (Tuple[int, int, int]): stride of the convolution.
        padding (Tuple[int, int, int]): padding of the convolution.
        dilation (Tuple[int, int, int]): dilation rate.
        algorithm: algorithm to use for the convolution. Defaults to global config.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, SparseConv3dNeighborCache]:
            - out_feats (torch.Tensor): [M, Co] tensor of output features.
            - out_coords (torch.Tensor): [M, 4] tensor of output coordinates.
            - neighbor_cache (SparseConv3dNeighborCache): neighbor cache for this operation.
    """
    Co, Kw, Kh, Kd, Ci = weight.shape
    kernel_size = (Kw, Kh, Kd)
    needs_grad = feats.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)

    is_out_coords_given = out_coords is not None
    if out_coords is None:
        out_coords = _get_output_coords(coords, shape, kernel_size, stride, padding, dilation)

    # NOTE:
    # In backward pass, workload reordering (sort by kernel mask) is only
    # necessary when the kernel mask distribution is highly irregular.
    #
    # For stride == 1 and auto-generated out_coords:
    #   - Almost all kernel offsets are valid for each input voxel
    #   - Bwd kernel masks are nearly all valid
    #   - Reordering is not necessary, use IGEMM
    #
    # For stride > 1 or user-provided out_coords:
    #   - Kernel masks vary significantly across voxels
    #   - Sorting by kernel mask greatly improves warp-level efficiency
    needs_bwd_sort = any(s != 1 for s in stride) or is_out_coords_given

    if neighbor_cache is None:
        neighbor_cache = _compute_neighbor_cache(
            coords, out_coords, shape, kernel_size, stride, padding, dilation,
            needs_grad, needs_bwd_sort=needs_bwd_sort,
        )

    SparseConv3dFunc = _select_spconv_function(algorithm)
    output, neighbor_cache = SparseConv3dFunc.apply(
        feats, neighbor_cache, weight.reshape(Co, Kw * Kh * Kd, Ci), bias
    )
    return output, out_coords, neighbor_cache

