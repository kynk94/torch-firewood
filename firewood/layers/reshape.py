import math
from typing import Optional, Tuple, Union, cast, overload

import torch.nn as nn
from torch import Size, Tensor

from firewood.common.types import INT


class Reshape(nn.Module):
    @overload
    def __init__(self, *, size: Optional[Union[Size, INT]]) -> None:
        ...

    @overload
    def __init__(self, *size: int) -> None:
        ...

    def __init__(
        self, *__size: int, size: Optional[Union[Size, INT]] = None
    ) -> None:
        super().__init__()
        if size is None:
            if not __size:
                raise ValueError("At least one dimension must be specified.")
            if len(__size) == 1 and isinstance(__size[0], tuple):
                size = __size[0]
            else:
                size = __size
        elif isinstance(size, int):
            size = (size,)
        self.size = tuple(size)

    def forward(self, input: Tensor) -> Tensor:
        return input.view(input.size(0), *self.size)


class _ReshapeNd(Reshape):
    rank: int
    channels: Optional[int] = None

    @overload
    def __init__(
        self, *, size: Union[Size, INT], channels: Optional[int]
    ) -> None:
        ...

    @overload
    def __init__(self, *size: int, channels: Optional[int]) -> None:
        ...

    def __init__(
        self,
        *__size: int,
        size: Optional[Union[Size, INT]] = None,
        channels: Optional[int] = None
    ) -> None:
        if not __size and size is None and channels is not None:
            # Assign value of `channels` to `size` to avoid exceptions.
            size = channels
        super().__init__(*__size, size=size)
        if self.size and channels and self.size[0] != channels:
            raise ValueError(
                "The number of channels must be equal to the first dimension "
                "of the size."
            )

        if channels:
            self.size = None  # type: ignore
            self.channels = channels

    def forward(self, input: Tensor) -> Tensor:
        if self.size is not None:
            return super().forward(input=input)

        # only works with square or cube Tensor
        input_dim = math.prod(input.shape[1:])
        spatial_dim = round((input_dim / self.channels) ** (1 / self.rank))
        return input.view(
            input.size(0), self.channels, *(spatial_dim,) * self.rank
        )


class Reshape1d(_ReshapeNd):
    rank = 1


class Reshape2d(_ReshapeNd):
    rank = 2


class Reshape3d(_ReshapeNd):
    rank = 3
