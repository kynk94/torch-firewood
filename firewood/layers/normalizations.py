import math
from typing import Any, Optional, Union, cast, overload

import numpy as np
import torch
import torch.linalg as LA
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from firewood.common.types import INT


@overload
def get(
    normalization: Union[nn.Module, str],
    num_features: int,
    eps: float,
    unbiased: bool,
    **normalization_kwargs: Any,
) -> nn.Module:
    ...


@overload
def get(
    normalization: None,
    num_features: int,
    eps: float,
    unbiased: bool,
    **normalization_kwargs: Any,
) -> None:
    ...


def get(
    normalization: Optional[Union[nn.Module, str]],
    num_features: int,
    eps: float = 1e-9,
    unbiased: bool = False,
    **kwargs: Any,
) -> Optional[nn.Module]:
    if normalization is None:
        return None
    if not isinstance(normalization, str):
        if not issubclass(normalization, nn.Module):  # type: ignore
            raise TypeError(
                "normalization must be a subclass of nn.Module or a string. "
                f"Got {type(normalization)}"
            )
        return normalization(num_features, **kwargs)
    normalization = normalization.lower()
    if normalization in {"bn", "batch", "batch_norm", "batch_normalization"}:
        return BatchNorm(num_features=num_features, eps=eps, **kwargs)
    if normalization in {
        "sbn",
        "sync_bn",
        "sync_batch",
        "sync_batch_norm",
        "sync_batch_normalization",
    }:
        return nn.SyncBatchNorm(num_features=num_features, eps=eps, **kwargs)
    if normalization in {"gn", "group", "group_norm", "group_normalization"}:
        return GroupNorm(num_channels=num_features, eps=eps, **kwargs)
    if normalization in {
        "in",
        "instance",
        "instance_norm",
        "instance_normalization",
    }:
        return InstanceNorm(
            num_features=num_features, eps=eps, unbiased=unbiased, **kwargs
        )
    if normalization in {
        "pn",
        "pixel",
        "pixel_norm",
        "pixel_normalization",
    }:
        return PixelNorm(eps=eps, **kwargs)
    raise ValueError(f"Unknown normalization: {normalization}")


class BatchNorm(_BatchNorm):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-9,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def _check_input_dim(self, input: Tensor) -> None:
        return


# subclass for ordering input arguments, and set default value of num_groups as
# half of num_channels.
class GroupNorm(nn.GroupNorm):
    def __init__(
        self,
        num_channels: int,
        num_groups: int = None,
        eps: float = 1e-9,
        affine: bool = True,
    ) -> None:
        if num_groups is None:
            num_groups = num_channels // 2
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )


class InstanceNorm(_InstanceNorm):
    """
    InstanceNorm of implicit input dimension.
    Does not support no_batch_dim operation.

    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-9,
        momentum: float = 0.1,
        unbiased: bool = False,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.unbiased = unbiased
        self.__fake_no_batch_dim = 3  # rank + 1, default is rank 2

    def _check_input_dim(self, input: Tensor) -> None:
        self.__fake_no_batch_dim = input.dim() - 1
        return

    def _get_no_batch_dim(self) -> int:
        return self.__fake_no_batch_dim

    def forward(self, input: Tensor) -> Tensor:
        # default operation does not support bessel correction
        if not self.unbiased:
            return super().forward(input)

        var, mean = torch.var_mean(
            input=input,
            dim=tuple(range(2, input.ndim)),
            unbiased=True,
            keepdim=True,
        )
        std = var.add(self.eps).sqrt()
        output = input.sub(mean).div(std)
        if self.affine:
            output = output.mul(self.weight).add(self.bias)
        return output

    def extra_repr(self) -> str:
        return super().extra_repr() + f", unbiased={self.unbiased}"


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        return moment_normalization(input, ord=2, dim=1, eps=self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


def moment_normalization(
    input: Tensor,
    ord: int = 2,
    dim: Optional[INT] = None,
    use_scaling: bool = True,
    eps: float = 1e-9,
) -> Tensor:
    """
    Return:
        input / (mean(input ** ord, dim) ** (1 / ord) + eps)
    """
    norm = LA.vector_norm(input, ord=ord, dim=dim, keepdim=True)
    if not use_scaling:
        return input / (norm + eps)

    if dim is None:
        dim = range(input.ndim)
    if isinstance(dim, int):
        numel = input.size(dim)
    else:
        numel = cast(int, np.prod(tuple(input.size(d) for d in dim)))
    return input / (norm / math.sqrt(numel) + eps)


def maximum_normalization(
    input: Tensor,
    dim: Optional[INT] = None,
    use_scaling: bool = True,
    eps: float = 1e-9,
) -> Tensor:
    """
    Return:
        input / (max(abs(input), except dim) + eps)
    """
    norm = LA.vector_norm(input, ord=float("inf"), dim=dim, keepdim=True)
    if not use_scaling:
        return input / (norm + eps)

    if dim is None:
        dim = range(input.ndim)
    if isinstance(dim, int):
        numel = input.size(dim)
    else:
        numel = cast(int, np.prod(tuple(input.size(d) for d in dim)))
    return input / (norm / math.sqrt(numel) + eps)
