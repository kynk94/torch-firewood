from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


def get(
    normalization: Optional[Union[nn.Module, str]],
    num_features: int,
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
        return BatchNorm(num_features=num_features, **kwargs)
    if normalization in {
        "sbn",
        "sync_bn",
        "sync_batch",
        "sync_batch_norm",
        "sync_batch_normalization",
    }:
        return nn.SyncBatchNorm(num_features=num_features, **kwargs)
    if normalization in {"gn", "group", "group_norm", "group_normalization"}:
        return GroupNorm(num_channels=num_features, **kwargs)
    if normalization in {
        "in",
        "instance",
        "instance_norm",
        "instance_normalization",
    }:
        return InstanceNorm(
            num_features=num_features, unbiased=unbiased, **kwargs
        )
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
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-9,
        momentum: float = 0.1,
        unbiased: bool = False,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        assert not (
            unbiased & affine
        ), "unbiased and affine cannot be True at the same time"
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.unbiased = unbiased

    def _check_input_dim(self, input: Tensor) -> None:
        return

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
        std = (var + self.eps).sqrt()
        return (input - mean) / std

    def extra_repr(self) -> str:
        return super().extra_repr() + f", unbiased={self.unbiased}"
