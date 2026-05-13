from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseUpsample:
    # TODO
    def __init__(self, size: tuple[int, ...], scale_factor: tuple[int, ...], mode: Literal["nearest", "linear"] = "nearest"):
        ...