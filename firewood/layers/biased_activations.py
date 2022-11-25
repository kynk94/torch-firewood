import functools
from typing import Any, Callable, Dict, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from firewood.common.constant import NULL_TENSOR
from firewood.common.types import DEVICE
from firewood.layers import activations
from firewood.utils.extensions import CUDAExtension

FORCE_DEFAULT = False


def _linear(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return x


def _relu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return F.relu(x)


def _leaky_relu(x: Tensor, alpha: float, *args: Any, **kwargs: Any) -> Tensor:
    return F.leaky_relu(x, negative_slope=alpha)


def _tanh(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return torch.tanh(x)


def _sigmoid(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return torch.sigmoid(x)


def _elu(x: Tensor, alpha: float, *args: Any, **kwargs: Any) -> Tensor:
    return F.elu(x, alpha=alpha)


def _selu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return F.selu(x)


def _softplus(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return F.softplus(x)


def _silu(x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    return F.silu(x)


class ACTIVATION(TypedDict):
    func: Callable
    default_alpha: float
    default_gain: Optional[float]
    cuda_operation_index: int
    gradient_reference: str
    has_second_grad: bool


# If default_gain is None, then nn.init.calculate_gain() is used.
ACTIVATIONS: Dict[str, ACTIVATION] = {
    "linear": {
        "func": _linear,
        "default_alpha": 1.0,
        "default_gain": 1,
        "cuda_operation_index": 1,
        "gradient_reference": "",
        "has_second_grad": False,
    },
    "relu": {
        "func": _relu,
        "default_alpha": 1.0,
        "default_gain": None,
        "cuda_operation_index": 2,
        "gradient_reference": "y",
        "has_second_grad": False,
    },
    "leaky_relu": {
        "func": _leaky_relu,
        "default_alpha": 0.2,
        # Pytorch implementation does not uses sqrt(2) as gain.
        "default_gain": np.sqrt(2),
        "cuda_operation_index": 3,
        "gradient_reference": "y",
        "has_second_grad": False,
    },
    "tanh": {
        "func": _tanh,
        "default_alpha": 1.0,
        # Mostly, tanh is used to fit the image in the range of [-1, 1],
        # so gain is defined as 1.
        "default_gain": 1,
        "cuda_operation_index": 4,
        "gradient_reference": "y",
        "has_second_grad": True,
    },
    "sigmoid": {
        "func": _sigmoid,
        "default_alpha": 1.0,
        "default_gain": 1,
        "cuda_operation_index": 5,
        "gradient_reference": "y",
        "has_second_grad": True,
    },
    "elu": {
        "func": _elu,
        "default_alpha": 1.0,
        "default_gain": 1,
        "cuda_operation_index": 6,
        "gradient_reference": "y",
        "has_second_grad": True,
    },
    "selu": {
        "func": _selu,
        # Does not use this alpha, use defined in cuda code.
        "default_alpha": 1.0,
        "default_gain": None,
        "cuda_operation_index": 7,
        "gradient_reference": "y",
        "has_second_grad": True,
    },
    "softplus": {
        "func": _softplus,
        "default_alpha": 1.0,
        "default_gain": 1,
        "cuda_operation_index": 8,
        "gradient_reference": "y",
        "has_second_grad": True,
    },
    "silu": {
        "func": _silu,
        "default_alpha": 1.0,
        "default_gain": np.sqrt(2),
        "cuda_operation_index": 9,
        "gradient_reference": "x",
        "has_second_grad": True,
    },
}


class BiasedActivation(nn.Module):
    force_default = FORCE_DEFAULT

    def __init__(
        self,
        activation: str,
        alpha: Optional[float] = None,
        gain: Optional[float] = None,
        clamp: Optional[float] = None,
        bias_add_dim: int = 1,
        device: Optional[DEVICE] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")

        self.activation = activations.normalize_activation_name(activation)
        self.bias_add_dim = bias_add_dim

        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation {self.activation}")
        activation_dict = ACTIVATIONS[self.activation]
        if alpha is None:
            self.alpha = activation_dict["default_alpha"]
        else:
            self.alpha = alpha
        if gain is None:
            gain = activation_dict.get("default_gain")
            if gain is None:
                gain = nn.init.calculate_gain(self.activation, self.alpha)
        self.gain = gain
        self.clamp = clamp if clamp is not None else -1

        self.register_parameter("bias", None)
        self.to(device=self.device)

    def _apply(self, fn: Callable[..., Any]) -> "BiasedActivation":
        if "t" in fn.__code__.co_varnames:
            with torch.no_grad():
                device = getattr(fn(NULL_TENSOR), "device", "cpu")
            self.device = torch.device(device)
        return super()._apply(fn)

    @property
    def operation(self) -> Callable[..., Tensor]:
        if self.device.type == "cpu" or self.force_default:
            return functools.partial(
                biased_activation,
                activation_function=ACTIVATIONS[self.activation]["func"],
                alpha=self.alpha,
                gain=self.gain,
                clamp=self.clamp,
                bias_add_dim=self.bias_add_dim,
            )
        try:
            return load_cuda_biased_activation(
                activation=self.activation,
                alpha=self.alpha,
                gain=self.gain,
                clamp=self.clamp,
                bias_add_dim=self.bias_add_dim,
            )
        except RuntimeError:
            self.force_default = True
            return self.operation

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:
        return self.operation(input, self.bias)

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"activation={self.activation}",
                f"alpha={self.alpha}",
                f"gain={self.gain:.04f}",
                f"clamp={self.clamp}",
                f"bias_add_dim={self.bias_add_dim}",
            ]
        )


def biased_activation(
    input: Tensor,
    bias: Optional[Tensor] = None,
    activation_function: Callable[..., Tensor] = _linear,
    alpha: float = 0.2,
    gain: float = 1.0,
    clamp: float = -1.0,
    bias_add_dim: int = 1,
) -> Tensor:
    output = input

    if bias is not None:
        output = output + bias.view(
            [-1 if i == bias_add_dim else 1 for i in range(output.ndim)]
        )
    output = activation_function(output, alpha=alpha)

    if gain != 1.0:
        output = output * gain

    if clamp >= 0:
        output = torch.clamp(output, -clamp, clamp)
    return output


def load_cuda_biased_activation(
    activation: str,
    alpha: float,
    gain: float = 1.0,
    clamp: float = -1.0,
    bias_add_dim: int = 1,
) -> Callable[..., Tensor]:
    name = "bias_act"
    cache_key = (
        activation,
        alpha,
        gain,
        clamp,
        bias_add_dim,
    )
    cache = CUDAExtension.cache(name)
    if cache_key in cache:
        return cache[cache_key].apply

    cuda_extension = CUDAExtension.get(name)

    cuda_operation_index = ACTIVATIONS[activation]["cuda_operation_index"]
    gradient_reference = ACTIVATIONS[activation]["gradient_reference"]
    has_second_grad = ACTIVATIONS[activation]["has_second_grad"]

    is_skip = activation == "linear" and gain == 1.0 and clamp < 0

    class BiasedActivationCUDA(torch.autograd.Function):
        @staticmethod
        # type: ignore[override]
        def forward(
            ctx: Any, input: Tensor, bias: Optional[Tensor] = None
        ) -> Tensor:
            if input.ndim > 2 and input.stride(1) == 1:
                ctx.memory_format = torch.channels_last
            else:
                ctx.memory_format = torch.contiguous_format

            input = input.contiguous(memory_format=ctx.memory_format)
            if bias is None:
                bias = NULL_TENSOR
            else:
                bias = bias.to(input.dtype).contiguous()

            if is_skip and bias is NULL_TENSOR:
                output = input
            else:
                output = cuda_extension(
                    x=input,
                    b=bias,
                    xref=NULL_TENSOR,
                    yref=NULL_TENSOR,
                    dy=NULL_TENSOR,
                    grad=0,
                    dim=bias_add_dim,
                    act=cuda_operation_index,
                    alpha=alpha,
                    gain=gain,
                    clamp=clamp,
                )

            if has_second_grad or gradient_reference == "x":
                memorized_input = input
                memorized_bias = bias
            else:
                memorized_input = NULL_TENSOR
                memorized_bias = NULL_TENSOR
            if gradient_reference == "y":
                memorized_output = output
            else:
                memorized_output = NULL_TENSOR
            ctx.save_for_backward(
                memorized_input, memorized_bias, memorized_output
            )
            return output

        @staticmethod
        # type: ignore[override]
        def backward(
            ctx: Any, output_gradient: Tensor
        ) -> Tuple[Tensor, Optional[Tensor]]:
            output_gradient = output_gradient.contiguous(
                memory_format=ctx.memory_format
            )
            input, bias, output = ctx.saved_tensors

            if (
                not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1])
                or is_skip
            ):
                input_gradient = output_gradient
            else:
                input_gradient = BiasedActivationGradCUDA.apply(
                    output_gradient, input, bias, output
                )

            if not ctx.needs_input_grad[1]:
                bias_gradient = None
            else:
                bias_gradient = input_gradient.sum(
                    [i for i in range(input_gradient.ndim) if i != bias_add_dim]
                )

            return input_gradient, bias_gradient

    class BiasedActivationGradCUDA(torch.autograd.Function):
        @staticmethod
        # type: ignore[override]
        def forward(
            ctx: Any,
            output_gradient: Tensor,
            input: Tensor,
            bias: Tensor,
            output: Tensor,
        ) -> Tensor:
            if output_gradient.ndim > 2 and output_gradient.stride(1) == 1:
                ctx.memory_format = torch.channels_last
            else:
                ctx.memory_format = torch.contiguous_format

            input_gradient = cuda_extension(
                x=output_gradient,
                b=bias,
                xref=input,
                yref=output,
                dy=NULL_TENSOR,
                grad=1,
                dim=bias_add_dim,
                act=cuda_operation_index,
                alpha=alpha,
                gain=gain,
                clamp=clamp,
            )

            ctx.save_for_backward(
                output_gradient if has_second_grad else NULL_TENSOR,
                input,
                bias,
                output,
            )
            return input_gradient

        @staticmethod
        # type: ignore[override]
        def backward(
            ctx: Any, input_second_gradient: Tensor
        ) -> Tuple[Any, Any, Any, None]:
            input_second_gradient = input_second_gradient.contiguous(
                memory_format=ctx.memory_format
            )
            output_gradient, input, bias, output = ctx.saved_tensors
            if not ctx.needs_input_grad[0]:
                output_second_gradient = None
            else:
                output_second_gradient = BiasedActivationGradCUDA.apply(
                    input_second_gradient, input, bias, output
                )

            if not has_second_grad:
                input_gradient = None
                bias_gradient = None
            else:
                if not (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]):
                    input_gradient = None
                else:
                    input_gradient = cuda_extension(
                        x=input_second_gradient,
                        b=bias,
                        xref=input,
                        yref=output,
                        dy=output_gradient,
                        grad=2,
                        dim=bias_add_dim,
                        act=cuda_operation_index,
                        alpha=alpha,
                        gain=gain,
                        clamp=clamp,
                    )

                if input_gradient is None or not ctx.needs_input_grad[2]:
                    bias_gradient = None
                else:
                    bias_gradient = input_gradient.sum(
                        [
                            i
                            for i in range(input_gradient.ndim)
                            if i != bias_add_dim
                        ]
                    )

            return (
                output_second_gradient,
                input_gradient,
                bias_gradient,
                None,
            )

    cache[cache_key] = BiasedActivationCUDA
    return BiasedActivationCUDA.apply
