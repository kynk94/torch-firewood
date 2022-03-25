from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Size, Tensor

from firewood.layers import initializers


def get(name: Optional[str], **kwargs: Any) -> Optional[nn.Module]:
    if name is None:
        return None
    name = name.lower()
    if name in {"normal", "gaussian"}:
        return GaussianNoise(**kwargs)
    if name == "uniform":
        return UniformNoise(**kwargs)
    raise ValueError(f"Unknown noise type: {name}")


class _NoiseBase(nn.Module):
    def __init__(
        self, strength: float = 0.0, channel_same: bool = True
    ) -> None:
        super().__init__()
        self.init_strength = strength
        self.channel_same = channel_same
        self.weight = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initializer = initializers.get("constant")
        initializer(self.weight, self.init_strength)

    def _get_noise_shape(self, input: Tensor) -> Size:
        input_shape = input.shape
        if self.channel_same:
            return Size((input_shape[0], 1, *input_shape[2:]))
        return input_shape

    def _generate_noise(self, shape: Union[Size, Tuple[int, ...]]) -> Tensor:
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        noise = self._generate_noise(self._get_noise_shape(input))
        weighted_noise = self.weight * noise.to(device=input.device)
        return input + weighted_noise.to(input.dtype)

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"channel_same={self.channel_same}",
                f"init_strength={self.init_strength}",
            ]
        )


class GaussianNoise(_NoiseBase):
    def __init__(
        self,
        stddev: float = 1.0,
        strength: float = 0.0,
        channel_same: bool = True,
    ) -> None:
        super().__init__(strength=strength, channel_same=channel_same)
        self.stddev = stddev

    def _generate_noise(self, shape: Union[Size, Tuple[int, ...]]) -> Tensor:
        return torch.normal(mean=0.0, std=self.stddev, size=shape)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", stddev={self.stddev}"


class UniformNoise(_NoiseBase):
    def __init__(
        self,
        min: float = -1.0,
        max: float = 1.0,
        strength: float = 0.0,
        channel_same: bool = True,
    ) -> None:
        super().__init__(strength=strength, channel_same=channel_same)
        self.min = min
        self.max = max

    def _generate_noise(self, shape: Union[Size, Tuple[int, ...]]) -> Tensor:
        return torch.rand(shape) * (self.max - self.min) + self.min

    def extra_repr(self) -> str:
        return super().extra_repr() + ", ".join(
            [
                "",
                f"min={self.min}",
                f"max={self.max}",
            ]
        )
