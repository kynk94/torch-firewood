from typing import Any, Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch._C import Graph, Value

from firewood.layers import initializers

_NORMAL_CACHE: Dict[
    Tuple[
        Union[Size, Tuple[int, ...]],  # shape
        float,  # mean
        float,  # stddev
        int,  # seed
    ],
    Type[torch.autograd.Function],
] = dict()


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
        self, strength: float = 1.0, channel_same: bool = True
    ) -> None:
        super().__init__()
        self.init_strength = strength
        self.channel_same = channel_same
        self.weight = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        initializer = initializers.get("constant")
        initializer(self.weight, self.init_strength)

    def _get_noise_shape(self, input: Tensor) -> Tuple[int, ...]:
        input_shape = input.shape
        if self.channel_same:
            return (input_shape[0], 1, *input_shape[2:])
        return tuple(input_shape)

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
        return _onnx_support_normal(shape, 0.0, self.stddev, seed=0)()

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


def _onnx_support_normal(
    shape: Union[Size, Tuple[int, ...]],
    mean: float = 0.0,
    stddev: float = 1.0,
    seed: int = 0,
) -> None:
    """
    Implementation for onnx support.

    This is implemented because `torch.normal` does not support onnx export.
    `torch.randn` supports onnx export, but slower than `torch.normal`
    """
    if not isinstance(shape, tuple):
        shape = tuple(shape)
    cache_key = (shape, mean, stddev, seed)
    if cache_key in _NORMAL_CACHE:
        return _NORMAL_CACHE[cache_key].apply

    class Normal(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any) -> Tensor:
            noise = torch.normal(mean=mean, std=stddev, size=shape)
            return noise

        @staticmethod
        def symbolic(g: Graph) -> Value:
            """
            After export, noise becomes a constant in onnx graph
            """
            g_mean = g.op("Constant", value_t=torch.tensor(mean))
            g_stddev = g.op("Constant", value_t=torch.tensor(stddev))
            noise = g.op("RandomNormal", shape_i=shape, seed_f=seed)
            moved_noise = g.op("Add", g.op("Mul", noise, g_stddev), g_mean)
            return moved_noise

        @staticmethod
        def backward(ctx: Any, grad_output: Tensor) -> Tensor:
            return grad_output

    _NORMAL_CACHE[cache_key] = Normal
    return Normal.apply
