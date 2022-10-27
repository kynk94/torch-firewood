from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

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
    noise: Optional[Tensor]

    def __init__(
        self,
        strength: float = 0.0,
        channel_same: bool = True,
        freeze_noise: bool = False,
    ) -> None:
        super().__init__()
        self.init_strength = strength
        self.channel_same = channel_same
        self.freeze_noise = freeze_noise
        self.weight = nn.Parameter(torch.zeros(1))
        self.register_buffer("noise", None)
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

    def reset_noise(self) -> None:
        self.noise = None

    def forward(self, input: Tensor) -> Tensor:
        if self.freeze_noise:
            if self.noise is None:
                self.noise = self._generate_noise(
                    self._get_noise_shape(input[:1])
                )
            noise = self.noise.expand(input.size(0), *(-1,) * (input.ndim - 1))
        else:
            noise = self._generate_noise(self._get_noise_shape(input))
        weight_size = (1, -1, *(1,) * (input.ndim - 2))
        weighted_noise = self.weight.view(weight_size) * noise.to(
            device=input.device
        )
        return input + weighted_noise.to(input.dtype)

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"channel_same={self.channel_same}",
                f"init_strength={self.init_strength}",
                f"freeze_noise={self.freeze_noise}",
            ]
        )


class GaussianNoise(_NoiseBase):
    def __init__(
        self,
        stddev: float = 1.0,
        strength: float = 0.0,
        channel_same: bool = True,
        freeze_noise: bool = False,
    ) -> None:
        super().__init__(
            strength=strength,
            channel_same=channel_same,
            freeze_noise=freeze_noise,
        )
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
        freeze_noise: bool = False,
    ) -> None:
        super().__init__(
            strength=strength,
            channel_same=channel_same,
            freeze_noise=freeze_noise,
        )
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
) -> Callable[..., Tensor]:
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
        # type: ignore[override]
        def forward(ctx: Any) -> Tensor:
            noise = torch.normal(mean=mean, std=stddev, size=shape)
            return noise

        @staticmethod
        def symbolic(g: Graph) -> Value:
            """
            After export, noise becomes a constant in onnx graph
            """
            noise = g.op("RandomNormal", shape_i=shape, seed_f=seed)  # type: ignore
            if stddev != 1.0:
                stddev = torch.tensor(stddev, dtype=torch.float32)
                g_stddev = g.op("Constant", value_t=stddev)  # type: ignore
                noise = g.op("Mul", noise, g_stddev)  # type: ignore
            if mean != 0.0:
                mean = torch.tensor(mean, dtype=torch.float32)
                g_mean = g.op("Constant", value_t=mean)  # type: ignore
                noise = g.op("Add", noise, g_mean)  # type: ignore
            return noise

        @staticmethod
        # type: ignore[override]
        def backward(ctx: Any, grad_output: Tensor) -> Tensor:
            return grad_output

    _NORMAL_CACHE[cache_key] = Normal
    return Normal.apply
