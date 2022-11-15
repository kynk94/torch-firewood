"""
Weight Denormalization introduced in StyleGAN2.

`weight_denorm(conv)` makes the following changes to the `conv` module.
before:
    output = weight * input + bias
after:
    module.use_extra_inputs = True  # The extra input is modulation_features.
    modulated_weight = weight * affine(modulation_features)
    demodulation_coeff = 1 / LA.vector_norm(modulated_weight)
    output = modulated_weight * input * demodulation_coeff + bias
"""
from typing import Literal, Optional, Tuple, Union

import torch
import torch.linalg as LA
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from firewood import utils
from firewood.functional.normalizations import (
    maximum_normalization,
    moment_normalization,
)
from firewood.layers.block import Block
from firewood.layers.linear import Linear


class WeightDeNormOutput:
    def __init__(self, out_features: int, name: str = "bias") -> None:
        self.out_features = out_features
        self.param_name = self.name = name
        self.demodulation_coeff: Optional[Tensor] = None

    @staticmethod
    def apply(
        module: nn.Module, out_features: int, name: str = "bias"
    ) -> "WeightDeNormOutput":
        fn = WeightDeNormOutput(out_features=out_features, name=name)
        module.register_forward_hook(fn)  # type: ignore
        bias: Optional[Union[Tensor, Parameter]] = getattr(module, name, None)
        if isinstance(bias, Parameter):
            # Remove bias to avoid adding bias immediately after weighting.
            delattr(module, name)
            fn.param_name += "_orig"
            module.register_parameter(fn.param_name, bias)
            setattr(module, name, bias.detach())
        return fn

    def remove(self, module: nn.Module) -> None:
        bias: Optional[Tensor] = getattr(self, self.name, None)
        if bias is not None:
            utils.popattr(module, self.name, None)
            module.register_parameter(self.name, Parameter(bias.detach()))
        delattr(module, self.param_name)

    def __call__(
        self, module: nn.Module, input: Tensor, output: Tensor
    ) -> Tensor:
        if self.demodulation_coeff is not None:
            output = output * self.demodulation_coeff
        output = output.view(-1, self.out_features, *output.shape[2:])

        bias: Optional[Tensor] = getattr(self, self.name, None)
        if bias is None:
            return output

        setattr(self, self.name, None)
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
        pre_normalize: Optional[str] = None,
        eps: float = 1e-9,
    ):
        self.param_name = self.name = name
        self.demodulate = demodulate
        self.pre_normalize = _normalize_pre_normalize_arg(pre_normalize)
        self.eps = eps

    @staticmethod
    def apply(
        module: nn.Module,
        modulation_features: int,
        name: str = "weight",
        demodulate: bool = True,
        pre_normalize: Optional[str] = None,
        eps: float = 1e-9,
    ) -> "WeightDeNorm":
        fn = WeightDeNorm(name, demodulate, pre_normalize, eps)
        module.register_forward_pre_hook(fn)  # type: ignore
        setattr(module, "use_extra_inputs", True)
        if hasattr(module, "groups"):
            setattr(module, "groups_orig", module.groups)

        in_features, out_features = utils.get_in_out_features(module)
        gamma_affine = Linear(
            in_features=modulation_features,
            out_features=in_features,
            bias=True,
            bias_initializer="ones",
        )
        module.add_module("gamma_affine", gamma_affine)

        weight: Union[Tensor, Parameter] = getattr(module, name)
        if isinstance(weight, Parameter):
            delattr(module, name)
            fn.param_name += "_orig"
            module.register_parameter(fn.param_name, weight)
            setattr(module, name, weight.detach())

        fn.output_hook = WeightDeNormOutput.apply(module, out_features)
        return fn

    def remove(self, module: nn.Module) -> None:
        weight: Optional[Tensor] = getattr(module, self.param_name, None)
        if weight is not None:
            utils.popattr(module, self.name, None)
            module.register_parameter(self.name, Parameter(weight.detach()))
        delattr(module, self.param_name)
        utils.popattr(module, "use_extra_inputs", None)
        if hasattr(module, "groups_orig"):
            setattr(module, "groups", utils.popattr(module, "groups_orig"))
        self.output_hook.remove(module)

    def __weight_denorm_fused(
        self, module: nn.Module, weight: Tensor, gamma: Tensor, batch_size: int
    ) -> None:
        if self.demodulate:
            modulated_weight = _weight_modulation(weight, gamma)
            demodulation_coeff = _calc_demodulation_coeff(
                modulated_weight, fused=True, eps=self.eps
            )
            weight = (modulated_weight * demodulation_coeff).flatten(0, 1)
        utils.keep_setattr(module, "groups", batch_size)
        setattr(module, self.name, weight)

    def __weight_denorm_not_fused(
        self, module: nn.Module, weight: Tensor, gamma: Tensor
    ) -> None:
        if self.demodulate:
            modulated_weight = _weight_modulation(weight, gamma)
            demodulation_coeff = _calc_demodulation_coeff(
                modulated_weight, fused=False, eps=self.eps
            )
            # coeff should be multiplied to the output of the module
            setattr(self.output_hook, "demodulation_coeff", demodulation_coeff)
        setattr(module, self.name, weight)

    def __call__(
        self, module: nn.Module, inputs: Tuple[Tensor, Tensor]
    ) -> Tensor:
        if not isinstance(inputs, tuple):
            raise ValueError("Expected a tuple of input and modulation.")
        input, modulation_input = inputs[:2]
        weight: Tensor = getattr(module, self.param_name)
        gamma: Tensor = module.get_submodule("gamma_affine")(modulation_input)

        if self.demodulate:
            if (
                self.pre_normalize == "stylegan2"
                and input.dtype == torch.float16
            ):
                weight, gamma = _pre_normalize_stylegan2(weight, gamma)
            if self.pre_normalize == "stylegan3":
                weight, gamma = _pre_normalize_stylegan3(weight, gamma)

        if getattr(module, "groups_orig", None) == 1:
            self.__weight_denorm_fused(module, weight, gamma, input.size(0))
            input = input.view(1, -1, *input.shape[2:])
        else:
            self.__weight_denorm_not_fused(module, weight, gamma)
            input = input * utils.unsqueeze_view(gamma, -1, input.ndim - 2)

        bias: Optional[Parameter] = getattr(
            module, self.output_hook.param_name, None
        )
        if bias is not None:
            # delete module's bias, and assign bias to the output_hook.
            setattr(module, self.output_hook.name, None)
            setattr(self.output_hook, self.output_hook.name, bias)
        return input


def _normalize_pre_normalize_arg(
    pre_normalize: Optional[str] = None,
) -> Optional[Literal["stylegan2", "stylegan3"]]:
    if pre_normalize is None:
        return None
    if pre_normalize.endswith("2"):
        return "stylegan2"
    if pre_normalize.endswith("3") or pre_normalize in {
        "af",
        "alias_free",
    }:
        return "stylegan3"
    raise ValueError(f"Unknown pre_normalize: {pre_normalize}")


def _pre_normalize_stylegan2(
    weight: Tensor, gamma: Tensor
) -> Tuple[Tensor, Tensor]:
    weight = maximum_normalization(weight, dim=tuple(range(1, weight.ndim)))
    gamma = maximum_normalization(gamma, dim=1, use_scaling=False)
    return weight, gamma


def _pre_normalize_stylegan3(
    weight: Tensor, gamma: Tensor
) -> Tuple[Tensor, Tensor]:
    weight = moment_normalization(weight, dim=tuple(range(1, weight.ndim)))
    gamma = moment_normalization(gamma, dim=None)
    return weight, gamma


def _weight_modulation(weight: Tensor, gamma: Tensor) -> Tensor:
    """
    Args:
        weight: (out_channels, in_channels, *kernel_size)
        gamma: (batch_size, out_features)
            gamma's out_features is equal to weight's in_channels.
    """
    rank = weight.ndim - 2
    batch_size, out_features = gamma.shape[:2]
    gamma = gamma.view(batch_size, 1, out_features, *(1,) * rank)
    return weight.unsqueeze(0) * gamma


def _calc_demodulation_coeff(
    modulated_weight: Tensor, fused: bool = True, eps: float = 1e-9
) -> Tensor:
    """
    modulated_weight:
        weight sized (batch_size, out_features, in_features, *kernel_size)
    """
    batch_size = modulated_weight.size(0)
    rank = modulated_weight.ndim - 3
    return_shape = (batch_size, -1) + (1,) * (rank + int(fused))
    coeff: Tensor = 1 / (
        LA.vector_norm(
            modulated_weight, ord=2, dim=tuple(range(2, modulated_weight.ndim))
        )
        + eps
    )
    return coeff.view(return_shape)


def weight_denorm(
    module: nn.Module,
    modulation_features: int,
    name: str = "weight",
    demodulate: bool = True,
    pre_normalize: str = "stylegan2",
    eps: float = 1e-9,
) -> nn.Module:
    if isinstance(module, Block):
        return weight_denorm(
            module.layers["weighting"],
            modulation_features,
            name,
            demodulate,
            pre_normalize,
            eps,
        )
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightDeNorm):
            return module
    WeightDeNorm.apply(
        module=module,
        modulation_features=modulation_features,
        name=name,
        demodulate=demodulate,
        pre_normalize=pre_normalize,
        eps=eps,
    )
    return module


def weight_denorm_to_conv(
    module: nn.Module,
    modulation_features: int,
    name: str = "weight",
    demodulate: bool = True,
    pre_normalize: str = "stylegan2",
    eps: float = 1e-9,
) -> nn.Module:
    for submodule in module.modules():
        module_name = utils.get_name(submodule)
        if "conv" not in module_name.lower():
            continue
        if not isinstance(getattr(submodule, name, None), Tensor):
            continue
        weight_denorm(
            module=submodule,
            modulation_features=modulation_features,
            name=name,
            demodulate=demodulate,
            pre_normalize=pre_normalize,
            eps=eps,
        )
    return module


def remove_weight_denorm(module: nn.Module) -> nn.Module:
    for submodule in module.modules():
        for k, hook in submodule._forward_pre_hooks.items():
            if isinstance(hook, WeightDeNorm):
                # WeightDeNorm removes WeightDeNormOutput too.
                hook.remove(submodule)
                del submodule._forward_pre_hooks[k]
                break
    return module
