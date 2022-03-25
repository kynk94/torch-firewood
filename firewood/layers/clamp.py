import torch
import torch.nn as nn
from torch import Tensor


class Clamp(nn.Module):
    def __init__(self, min: float = -1, max: float = 1):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, input: Tensor) -> Tensor:
        return torch.clamp(input, self.min, self.max)

    def extra_repr(self) -> str:
        return f"min={self.min}, max={self.max}"
