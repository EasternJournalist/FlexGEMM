import torch
from torch import Tensor

def sparse_upsample(
    feats: Tensor,
    coords: Tensor,
    scale_factor: tuple[int, ...], 
    output_coords: Tensor | None = None,
    mode: Literal["nearest", "linear"] = "nearest",
    cache: IndexCache | NeighborCache | None = None
):
    ...# TODO
    
    


