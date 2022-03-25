import functools
import math
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch import linalg as LA
from torch.nn import Parameter
from torch.nn.utils import spectral_norm, weight_norm

from firewood import utils
from firewood.layers.linear import Linear


def get(
    normalization: str,
    n_power_iterations: int = 1,
    modulation_features: Optional[int] = None,
    demodulate: bool = True,
    pre_normalize: bool = False,
    eps: float = 1e-9,
    **kwargs: Any,
) -> Optional[Callable[..., nn.Module]]:
    if normalization is None:
        return None
    normalization = normalization.lower()
    if normalization.startswith("spectral"):
        return functools.partial(
            spectral_norm,
            n_power_iterations=n_power_iterations,
            eps=eps,
            **kwargs,
        )
    if normalization in {"weight", "weight_norm", "weight_normalization"}:
        return functools.partial(weight_norm, **kwargs)
    if normalization in {
        "demodulation",
        "weight_demodulation",
        "denorm",
        "weight_denorm",
        "weight_denormalization",
    }:
        if modulation_features is None:
            raise ValueError("modulation_features must be specified.")
        return functools.partial(
            weight_denorm,
            modulation_features=modulation_features,
            demodulate=demodulate,
            pre_normalize=pre_normalize,
            eps=eps,
            **kwargs,
        )
    raise ValueError(f"Unknown weight normalization: {normalization}")


class WeightDeNormOutput:
    def __init__(self, out_features: int, name: str = "bias") -> None:
        self.out_features = out_features
        self.name = name
        self.demodulation_coef: Tensor = None  # type: ignore

        self.target_name = self.name

    @staticmethod
    def apply(
        module: nn.Module, out_features: int, name: str = "bias"
    ) -> "WeightDeNormOutput":
        fn = WeightDeNormOutput(out_features=out_features, name=name)
        module.register_forward_hook(fn)  # type: ignore
        bias: Optional[Union[Tensor, Parameter]] = getattr(module, "bias", None)
        if isinstance(bias, Parameter):
            delattr(module, "bias")
            module.register_parameter(name + "_orig", Parameter(bias.data))
            setattr(fn, "target_name", name + "_orig")
            setattr(module, "bias", bias.data)
        return fn

    def remove(self, module: nn.Module) -> None:
        bias: Optional[Tensor] = getattr(self, self.name, None)
        if bias is not None:
            utils.popattr(module, self.name, None)
            module.register_parameter(self.name, Parameter(bias.detach()))
        delattr(module, self.target_name)

    def __call__(
        self, module: nn.Module, input: Tensor, output: Tensor
    ) -> Tensor:
        if self.demodulation_coef is not None:
            output = output * self.demodulation_coef
        output = output.view(-1, self.out_features, *output.shape[2:])

        bias: Optional[Tensor] = getattr(self, self.name, None)
        setattr(self, self.name, None)
        if bias is None:
            return output

        setattr(module, self.name, bias)
        bias = bias.view([-1 if i == 1 else 1 for i in range(output.ndim)])
        return output + bias


class WeightDeNorm:
    def __init__(
        self,
        name: str = "weight",
        demodulate: bool = True,
        pre_normalize: bool = False,
        eps: float = 1e-9,
    ):
        self.name = name
        self.demodulate = demodulate
        self.pre_normalize = pre_normalize
        self.eps = eps

        self.target_name = self.name
        self.output_hook: WeightDeNormOutput = None  # type: ignore

    @staticmethod
    def apply(
        module: nn.Module,
        modulation_features: int,
        name: str = "weight",
        demodulate: bool = True,
        pre_normalize: bool = False,
        eps: float = 1e-9,
    ) -> "WeightDeNorm":
        if getattr(module, "groups", 1) != 1:
            raise ValueError(
                "WeightDeNorm is not supported for weight_layer with groups > 1"
            )

        fn = WeightDeNorm(name, demodulate, pre_normalize, eps)
        module.register_forward_pre_hook(fn)  # type: ignore
        setattr(module, "use_external_input", True)

        in_features, out_features = utils.get_in_out_features(module)
        gamma_linear = Linear(
            in_features=modulation_features,
            out_features=in_features,
            bias=True,
            bias_initializer="ones",
        )
        module.add_module("gamma_linear", gamma_linear)

        weight: Union[Tensor, Parameter] = getattr(module, name)
        if isinstance(weight, Parameter):
            delattr(module, name)
            module.register_parameter(name + "_orig", Parameter(weight.data))
            setattr(fn, "target_name", name + "_orig")
            setattr(module, name, weight.data)

        setattr(
            fn, "output_hook", WeightDeNormOutput.apply(module, out_features)
        )
        return fn

    def remove(self, module: nn.Module) -> None:
        weight: Optional[Tensor] = getattr(module, self.target_name, None)
        if weight is not None:
            utils.popattr(module, self.name, None)
            module.register_parameter(self.name, Parameter(weight.detach()))
        delattr(module, self.target_name)
        self.output_hook.remove(module)

    def __call__(
        self, module: nn.Module, inputs: Tuple[Tensor, Tensor]
    ) -> Tensor:
        if not isinstance(inputs, tuple):
            raise ValueError("Expected a tuple of input and modulation.")
        input, modulation_input = inputs[:2]

        weight: Tensor = getattr(module, self.target_name)
        weight = weight.to(dtype=input.dtype)
        gamma: Tensor = getattr(module, "gamma_linear")(modulation_input)
        gamma = gamma.to(dtype=input.dtype)

        if self.demodulate:
            weight, gamma = self._pre_normalize(weight, gamma, input.dtype)
            modulated_weight = self._weight_modulation(weight, gamma)
            demodulation_coef = self._calc_demodulation_coef(modulated_weight)

        # fused operation
        if hasattr(module, "groups"):
            setattr(module, "groups", input.size(0))
            input = input.view(1, -1, *input.shape[2:])
            if self.demodulate:
                modulated_weight = torch.mul(
                    modulated_weight, demodulation_coef
                )
            weight = modulated_weight.view(-1, *modulated_weight.shape[2:])
        # non-fused operation
        else:
            input = input * gamma.view(*gamma.shape, *(1,) * (input.ndim - 2))

        setattr(module, self.name, weight)
        if not hasattr(module, "groups") and self.demodulate:
            if demodulation_coef.ndim > 2:
                demodulation_coef = demodulation_coef.squeeze(-1)
            setattr(self.output_hook, "demodulation_coef", demodulation_coef)

        bias = getattr(module, self.output_hook.target_name)
        if bias is not None:
            # delete module's bias, and assign bias to the forward hook
            setattr(module, self.output_hook.name, None)
            setattr(self.output_hook, self.output_hook.name, bias)
        return input

    def _pre_normalize(
        self,
        weight: Tensor,
        gamma: Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Tensor, Tensor]:
        # alias-free-gan option
        if self.pre_normalize:
            # root mean squared
            weight = (
                weight
                * weight.square()
                .mean(dim=tuple(range(1, weight.ndim)), keepdim=True)
                .rsqrt()
            )
            gamma = gamma * gamma.square().mean().rsqrt()
        # stylegan2 option, prevent overflow
        elif dtype == torch.float16:
            # maximum norm
            norm: Tensor = LA.vector_norm(
                weight,
                ord=float("inf"),
                dim=tuple(range(1, weight.ndim)),
                keepdim=True,
            )
            weight = weight / math.sqrt(weight[0].numel()) / norm
            gamma = gamma / LA.vector_norm(
                gamma, ord=float("inf"), dim=1, keepdim=True
            )
        return weight, gamma

    def _weight_modulation(self, weight: Tensor, gamma: Tensor) -> Tensor:
        rank = weight.ndim - 2
        batch_size = gamma.size(0)
        batch_size, out_features = gamma.shape[:2]
        gamma = gamma.view(batch_size, 1, out_features, *(1,) * rank)
        return torch.mul(weight.unsqueeze(0), gamma)

    def _calc_demodulation_coef(self, modulated_weight: Tensor) -> Tensor:
        batch_size = modulated_weight.size(0)
        rank = modulated_weight.ndim - 3
        demodulation_coef = torch.rsqrt(
            modulated_weight.square().sum(
                dim=tuple(range(2, modulated_weight.ndim))
            )
            + self.eps
        ).view(batch_size, -1, 1, *(1,) * rank)
        return demodulation_coef


def weight_denorm(
    module: nn.Module, modulation_features: int, **kwargs: Any
) -> nn.Module:
    WeightDeNorm.apply(module, modulation_features, **kwargs)
    return module


def remove_weight_denorm(module: nn.Module) -> nn.Module:
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightDeNorm):
            hook.remove(module)  # WeightDeNorm removes WeightDeNormOutput too.
            del module._forward_pre_hooks[k]
            break
    return module
