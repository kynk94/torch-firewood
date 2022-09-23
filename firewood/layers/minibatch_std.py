from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class MinibatchStd(nn.Module):
    """
    Extract minibatch standard deviation feature.

    Improved Techniques for Training GANs
    https://arxiv.org/abs/1606.03498
    """

    def __init__(
        self,
        size: int = 4,
        averaging: str = "all",
        concat: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if averaging not in {"all", "spatial", "channel"}:
            raise ValueError(f"Not supported averaging mode: {averaging}")
        self.size = size
        self.averaging = averaging
        self.concat = concat
        self.eps = eps

    def forward(self, input: Tensor, size: Optional[int] = None) -> Tensor:
        B = input.size(0)
        grouped_std = self.calc_grouped_std(input.float(), size)
        M, _, *S = grouped_std.shape
        repeats = (B // M, 1, *S)

        if self.averaging == "all":  # (M, 1, *1) after averaging
            dim = tuple(range(1, grouped_std.ndim))
        elif self.averaging == "spatial":  # (M, C, *1) after averaging
            dim = tuple(range(2, grouped_std.ndim))
        elif self.averaging == "channel":  # (M, 1, *S) after averaging
            dim = (1,)
            repeats = (*repeats[:2], *(1,) * len(S))
        else:
            raise ValueError(f"Not supported averaging mode: {self.averaging}")
        feature = grouped_std.mean(dim=dim, keepdim=True).to(dtype=input.dtype)

        if self.concat:
            return torch.cat((input, feature.repeat(repeats)), dim=1)
        return feature

    def calc_grouped_std(
        self, input: Tensor, size: Optional[int] = None
    ) -> Tensor:
        B, C, *S = input.shape
        size = size or self.size or max(1, B // 2)
        size = min(B, size)
        if B % size != 0:
            raise ValueError(
                f"The number of batch {B} is not divisible by size {size}"
            )
        grouped_input = input.view(size, -1, C, *S)
        variance = grouped_input.var(0, unbiased=False, keepdim=False)
        return variance.add(self.eps).sqrt()

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"size={self.size}",
                f"averaging={self.averaging}",
                f"concat={self.concat}",
                f"eps={self.eps}",
            ]
        )
