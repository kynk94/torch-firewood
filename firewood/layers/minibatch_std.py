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
        groups: int = 4,
        averaging: str = "all",
        concat: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if averaging not in {"all", "spatial", "channel"}:
            raise ValueError(f"Not supported averaging mode: {averaging}")
        self.groups = groups
        self.averaging = averaging
        self.concat = concat
        self.eps = eps

    def forward(self, input: Tensor, groups: Optional[int] = None) -> Tensor:
        B = input.size(0)
        grouped_std = self.calc_grouped_std(input.float(), groups)
        M = grouped_std.size(0)

        if self.averaging == "all":  # (M, 1, *1) after averaging
            dim = tuple(range(1, grouped_std.ndim))
        elif self.averaging == "spatial":  # (M, C, *1) after averaging
            dim = tuple(range(2, grouped_std.ndim))
        elif self.averaging == "channel":  # (M, 1, *S) after averaging
            dim = (1,)
        else:
            raise ValueError(f"Not supported averaging mode: {self.averaging}")
        feature = grouped_std.mean(dim=dim, keepdim=True).to(dtype=input.dtype)

        if not self.concat:
            return feature

        feature = feature.repeat(B // M, *(1,) * (feature.ndim - 1))
        return torch.cat((input, feature), dim=1)

    def calc_grouped_std(
        self, input: Tensor, groups: Optional[int] = None
    ) -> Tensor:
        B, C, *S = input.shape
        groups = groups or self.groups or max(1, B // 2)
        groups = min(B, groups)
        if B % groups != 0:
            raise ValueError(
                f"The number of batch {B} is not divisible by groups {groups}"
            )
        grouped_input = input.view(groups, -1, C, *S)
        variance = grouped_input.var(0, unbiased=False, keepdim=False)
        return variance.add(self.eps).sqrt()
