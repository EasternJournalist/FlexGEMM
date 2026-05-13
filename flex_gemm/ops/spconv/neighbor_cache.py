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


class SparseConvNeighborCache:

    symmetric: bool
    """ When True, input/output coordinates coincide and kernel offsets are
    centrally symmetric, so the backward-input pass can reuse the forward
    cache with the weight flipped along the V dimension."
    """

    num_input_coords: int
    "Number of input coordinates (rows of the bwd neighbor map)."

    num_output_coords: int
    "Number of output coordinates (rows of the fwd neighbor map)."

    def __init__(
        self,
        fwd_neighbor_map: Tensor | None = None,
        bwd_neighbor_map: Tensor | None = None,
        *,
        num_input_coords: int | None = None,
        num_output_coords: int | None = None,
        symmetric: bool = False,
    ):
        assert fwd_neighbor_map is not None or bwd_neighbor_map is not None, \
            "At least one of forward/backward neighbor map should be provided"

        self.symmetric = symmetric
        if symmetric:
            assert num_output_coords is None or num_input_coords is None or num_output_coords == num_input_coords, \
                "symmetric=True implies num_input_coords == num_output_coords"
            n = num_input_coords if num_input_coords is not None else num_output_coords
            if n is None:
                # Infer from whichever map was provided.
                n = (fwd_neighbor_map if fwd_neighbor_map is not None else bwd_neighbor_map).shape[0]
            self.num_input_coords = n
            self.num_output_coords = n
        else:
            assert num_input_coords is not None and num_output_coords is not None, \
                "Non-symmetric SparseConvNeighborCache requires both num_input_coords and num_output_coords"
            self.num_input_coords = num_input_coords
            self.num_output_coords = num_output_coords

        if fwd_neighbor_map is not None:
            assert fwd_neighbor_map.shape[0] == self.num_output_coords, \
                f"fwd_neighbor_map.shape[0]={fwd_neighbor_map.shape[0]} but num_output_coords={self.num_output_coords}"
            self['_fwd_neighbor_map'] = fwd_neighbor_map
        if bwd_neighbor_map is not None:
            assert bwd_neighbor_map.shape[0] == self.num_input_coords, \
                f"bwd_neighbor_map.shape[0]={bwd_neighbor_map.shape[0]} but num_input_coords={self.num_input_coords}"
            self['_bwd_neighbor_map'] = bwd_neighbor_map


    @abstractmethod
    def build_fwd_neighbor_map(self) -> Tensor:
        """Implement this method to build the forward neighbor map. Result will be cached"""
        raise NotImplementedError

    @abstractmethod
    def build_bwd_neighbor_map(self) -> Tensor:
        """Implement this method to build the backward neighbor map. Result will be cached"""
        raise NotImplementedError

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return hasattr(self, key)

    def _fwd_post_process_gray_code_sort(self) -> None:
        self['_fwd_gray_code'], self['_fwd_sorted_idx'] = \
            kernels.triton.neighbor_map_gray_code_sort(self.fwd_neighbor_mask)

    def _fwd_post_process_valid_signal(self) -> None:
        self['_fwd_valid_signal_i'], self['_fwd_valid_signal_o'], self['_fwd_valid_signal_seg'] = \
            kernels.triton.neighbor_map_valid_signal(self.fwd_neighbor_map, self.fwd_neighbor_mask)
            
    def _fwd_post_process_valid_kernel(self, block_size: int) -> None:
        self[f'_fwd_valid_kernel_{block_size}'], self[f'_fwd_valid_kernel_seg_{block_size}'] = \
            kernels.triton.neighbor_map_valid_kernel(self['_fwd_gray_code'], self['_fwd_sorted_idx'], block_size)

    @property
    def fwd_neighbor_map(self) -> Tensor:
        if '_fwd_neighbor_map' not in self:
            self['_fwd_neighbor_map'] = kernels.triton.transpose_neighbor_map_triton(
                self.bwd_neighbor_map, self.num_output_coords
            )
        return self['_fwd_neighbor_map']

    @property
    def fwd_neighbor_mask(self) -> Tensor:
        if '_fwd_neighbor_mask' not in self:
            self['_fwd_neighbor_mask'] = self.fwd_neighbor_map.view(dtype=torch.int32) != -1
        return self['_fwd_neighbor_mask']

    @property
    def fwd_gray_code(self) -> Tensor:
        if '_fwd_gray_code' not in self:
            self._fwd_post_process_gray_code_sort()
        return self['_fwd_gray_code']

    @property
    def fwd_sorted_idx(self) -> Tensor:
        if '_fwd_sorted_idx' not in self:
            self._fwd_post_process_gray_code_sort()
        return self['_fwd_sorted_idx']

    @property
    def fwd_valid_signal_i(self) -> Tensor:
        if '_fwd_valid_signal_i' not in self:
            self._fwd_post_process_valid_signal()
        return self['_fwd_valid_signal_i']

    @property
    def fwd_valid_signal_o(self) -> Tensor:
        if '_fwd_valid_signal_o' not in self:
            self._fwd_post_process_valid_signal()
        return self['_fwd_valid_signal_o']

    @property
    def fwd_valid_signal_seg(self) -> Tensor:
        if '_fwd_valid_signal_seg' not in self:
            self._fwd_post_process_valid_signal()
        return self['_fwd_valid_signal_seg']

    def fwd_valid_kernel_callback(self, block_size: int) -> Tensor:
        if f'_fwd_valid_kernel_{block_size}' not in self or f'_fwd_valid_kernel_seg_{block_size}' not in self:
            self._fwd_post_process_valid_kernel(block_size)
        return self[f'_fwd_valid_kernel_{block_size}']
    
    def fwd_valid_kernel_seg_callback(self, block_size: int) -> Tensor:
        if f'_fwd_valid_kernel_{block_size}' not in self or f'_fwd_valid_kernel_seg_{block_size}' not in self:
            self._fwd_post_process_valid_kernel(block_size)
        return self[f'_fwd_valid_kernel_seg_{block_size}']

    def _bwd_post_process_gray_code_sort(self) -> None:
        self['_bwd_gray_code'], self['_bwd_sorted_idx'] = \
            kernels.triton.neighbor_map_gray_code_sort(self.bwd_neighbor_mask)

    def _bwd_post_process_valid_signal(self) -> None:
        self['_bwd_valid_signal_i'], self['_bwd_valid_signal_o'], self['_bwd_valid_signal_seg'] = \
            kernels.triton.neighbor_map_valid_signal(self.bwd_neighbor_map, self.bwd_neighbor_mask)
            
    def _bwd_post_process_valid_kernel(self, block_size: int) -> None:
        self[f'_bwd_valid_kernel_{block_size}'], self[f'_bwd_valid_kernel_seg_{block_size}'] = \
            kernels.triton.neighbor_map_valid_kernel(self.bwd_gray_code, self.bwd_sorted_idx, block_size)

    @property
    def bwd_neighbor_map(self) -> Tensor:
        if '_bwd_neighbor_map' not in self:
            self['_bwd_neighbor_map'] = kernels.triton.transpose_neighbor_map_triton(
                self.fwd_neighbor_map, self.num_input_coords
            )
        return self['_bwd_neighbor_map']

    @property
    def bwd_neighbor_mask(self) -> Tensor:
        if '_bwd_neighbor_mask' not in self:
            self['_bwd_neighbor_mask'] = self.bwd_neighbor_map.view(dtype=torch.int32) != -1
        return self['_bwd_neighbor_mask']

    @property
    def bwd_gray_code(self) -> Tensor:
        if '_bwd_gray_code' not in self:
            self._bwd_post_process_gray_code_sort()
        return self['_bwd_gray_code']

    @property
    def bwd_sorted_idx(self) -> Tensor:
        if '_bwd_sorted_idx' not in self:
            self._bwd_post_process_gray_code_sort()
        return self['_bwd_sorted_idx']

    @property
    def bwd_valid_signal_i(self) -> Tensor:
        if '_bwd_valid_signal_i' not in self:
            self._bwd_post_process_valid_signal()
        return self['_bwd_valid_signal_i']

    @property
    def bwd_valid_signal_o(self) -> Tensor:
        if '_bwd_valid_signal_o' not in self:
            self._bwd_post_process_valid_signal()
        return self['_bwd_valid_signal_o']

    def bwd_valid_kernel_callback(self, block_size: int) -> Tensor:
        if f'_bwd_valid_kernel_{block_size}' not in self or f'_bwd_valid_kernel_seg_{block_size}' not in self:
            self._bwd_post_process_valid_kernel(block_size)
        return self[f'_bwd_valid_kernel_{block_size}']
    
    def bwd_valid_kernel_seg_callback(self, block_size: int) -> Tensor:
        if f'_bwd_valid_kernel_{block_size}' not in self or f'_bwd_valid_kernel_seg_{block_size}' not in self:
            self._bwd_post_process_valid_kernel(block_size)
        return self[f'_bwd_valid_kernel_seg_{block_size}']
            
