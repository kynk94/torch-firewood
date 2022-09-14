import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch import linalg as LA
from torch.nn import Parameter

from firewood import utils
from firewood.layers.linear import Linear

"""
WeightDeNorm is a hook that applies weight denormalization to a module.

Operations:
    1. Normalize the weight of the module.
    2. Apply the weight normalization to the module.
    3. Apply the weight denormalization to the module.
"""


class WeightDeNormOutput:
    def __init__(self, out_features: int, name: str = "bias") -> None:
        self.out_features = out_features
        self.call_name = self.name = name
        self.demodulation_coef: Optional[Tensor] = None

    @staticmethod
    def apply(
        module: nn.Module, out_features: int, name: str = "bias"
    ) -> "WeightDeNormOutput":
        fn = WeightDeNormOutput(out_features=out_features, name=name)
        module.register_forward_hook(fn)  # type: ignore
        bias: Optional[Union[Tensor, Parameter]] = getattr(module, "bias", None)
        if isinstance(bias, Parameter):
            delattr(module, "bias")
            module.register_parameter(name + "_orig", Parameter(bias.detach()))
            setattr(fn, "call_name", name + "_orig")
            setattr(module, "bias", bias.detach())
        return fn

    def remove(self, module: nn.Module) -> None:
        bias: Optional[Tensor] = getattr(self, self.name, None)
        if bias is not None:
            utils.popattr(module, self.name, None)
            module.register_parameter(self.name, Parameter(bias.detach()))
        delattr(module, self.call_name)

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
    """
    Weight demodulation operation introduced in StyleGAN2.

    Operate as forward-pre and forward hook.
    hook sequence:
        (LREqualizer) -> WeightDeNorm -> module's forward -> WeightDeNormOutput
    """

    output_hook: WeightDeNormOutput

    def __init__(
        self,
        name: str = "weight",
        demodulate: bool = True,
        pre_normalize: bool = False,
        eps: float = 1e-9,
    ):
        self.call_name = self.name = name
        self.demodulate = demodulate
        self.pre_normalize = pre_normalize
        self.eps = eps

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
            fn.call_name += "_orig"
            module.register_parameter(fn.call_name, weight)
            setattr(module, name, weight.detach())

        fn.output_hook = WeightDeNormOutput.apply(module, out_features)
        return fn

    def remove(self, module: nn.Module) -> None:
        weight: Optional[Tensor] = getattr(module, self.call_name, None)
        if weight is not None:
            utils.popattr(module, self.name, None)
            module.register_parameter(self.name, Parameter(weight.detach()))
        delattr(module, self.call_name)
        self.output_hook.remove(module)

    def __call__(
        self, module: nn.Module, inputs: Tuple[Tensor, Tensor]
    ) -> Tensor:
        if not isinstance(inputs, tuple):
            raise ValueError("Expected a tuple of input and modulation.")
        input, modulation_input = inputs[:2]

        weight: Tensor = getattr(module, self.call_name)
        weight = weight.to(dtype=input.dtype)
        gamma: Tensor = module.get_submodule("gamma_linear")(modulation_input)
        gamma = gamma.to(dtype=input.dtype)

        if self.demodulate:
            weight, gamma = self._pre_normalize(weight, gamma, input.dtype)
            modulated_weight = self._weight_modulation(weight, gamma)
            demodulation_coef = self._calc_demodulation_coef(modulated_weight)

        # fused operation
        original_groups = getattr(module, "groups", None)
        if original_groups is not None:
            setattr(module, "groups", input.size(0))
            setattr(module, "groups_orig", original_groups)
            input = input.view(1, -1, *input.shape[2:])
            if self.demodulate:
                modulated_weight = modulated_weight * demodulation_coef
            weight = modulated_weight.view(-1, *modulated_weight.shape[2:])
            original_weight_shape = getattr(module, "weight_shape", None)
            if original_weight_shape is not None:
                setattr(module, "weight_shape", tuple(weight.shape))
                setattr(module, "weight_shape_orig", original_weight_shape)
        # non-fused operation
        else:
            input = input * utils.unsqueeze_view(gamma, -1, input.ndim - 2)

        setattr(module, self.name, weight)
        if not hasattr(module, "groups") and self.demodulate:
            if demodulation_coef.ndim > 2:
                demodulation_coef = demodulation_coef.squeeze(-1)
            setattr(self.output_hook, "demodulation_coef", demodulation_coef)

        bias = getattr(module, self.output_hook.call_name)
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
