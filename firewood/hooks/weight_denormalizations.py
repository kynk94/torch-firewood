import math
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch import linalg as LA
from torch.nn import Parameter

from firewood import utils
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
        module.add_module("gamma_affine", gamma_linear)

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
        if hasattr(module, "groups_orig"):
            groups = utils.popattr(module, "groups_orig")
            setattr(module, "groups", groups)
        if hasattr(module, "weight_shape_orig"):
            weight_shape = utils.popattr(module, "weight_shape_orig")
            setattr(module, "weight_shape", weight_shape)
        self.output_hook.remove(module)

    def __weight_denorm_fused(
        self, module: nn.Module, weight: Tensor, gamma: Tensor, input: Tensor
    ) -> None:
        if self.demodulate:
            modulated_weight = _weight_modulation(weight, gamma)
            demodulation_coeff = _calc_demodulation_coeff(
                modulated_weight, fused=True, eps=self.eps
            )
            modulated_weight = modulated_weight * demodulation_coeff
        # grouped convolution
        utils.keep_setattr(module, "groups", input.size(0))
        input = input.view(1, -1, *input.shape[2:])
        weight = modulated_weight.flatten(0, 1)
        # to support GFixConv of `firewood.layers.conv_gradfix`
        if hasattr(module, "weight_shape"):
            utils.keep_setattr(module, "weight_shape", weight.shape)
        setattr(module, self.name, weight)

    def __weight_denorm_not_fused(
        self, module: nn.Module, weight: Tensor, gamma: Tensor, input: Tensor
    ) -> None:
        if self.demodulate:
            modulated_weight = _weight_modulation(weight, gamma)
            demodulation_coeff = _calc_demodulation_coeff(
                modulated_weight, fused=False, eps=self.eps
            )
            # coeff should be multiplied to the output of the module
            setattr(self.output_hook, "demodulation_coeff", demodulation_coeff)
        input = input * utils.unsqueeze_view(gamma, -1, input.ndim - 2)
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

        if getattr(module, "groups", None) is not None:
            self.__weight_denorm_fused(module, weight, gamma, input)
        else:
            self.__weight_denorm_not_fused(module, weight, gamma, input)

        bias = getattr(module, self.output_hook.param_name, None)
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
    # maximum norm
    norm: Tensor = LA.vector_norm(
        weight,
        ord=float("inf"),
        dim=tuple(range(1, weight.ndim)),
        keepdim=True,
    )
    weight = weight / math.sqrt(weight[0].numel()) / norm
    gamma = gamma / LA.vector_norm(gamma, ord=float("inf"), dim=1, keepdim=True)
    return weight, gamma


def _pre_normalize_stylegan3(
    weight: Tensor, gamma: Tensor
) -> Tuple[Tensor, Tensor]:
    # root mean squared
    weight = (
        weight
        * weight.square()
        .mean(dim=tuple(range(1, weight.ndim)), keepdim=True)
        .rsqrt()
    )
    gamma = gamma * gamma.square().mean().rsqrt()
    return weight, gamma


def _weight_modulation(weight: Tensor, gamma: Tensor) -> Tensor:
    rank = weight.ndim - 2
    batch_size, out_features = gamma.shape[:2]
    gamma = gamma.view(batch_size, 1, out_features, *(1,) * rank)
    return weight.unsqueeze(0) * gamma


def _calc_demodulation_coeff(
    modulated_weight: Tensor, fused: bool = True, eps: float = 1e-9
) -> Tensor:
    """
    modulated_weight:
        weight sized (batch_size, in_features, out_features, *spatial)
    """
    batch_size = modulated_weight.size(0)
    rank = modulated_weight.ndim - 3
    return_shape = (batch_size, -1, 1) + (1,) * (rank - int(not fused))
    return (
        modulated_weight.square()
        .sum(dim=tuple(range(2, modulated_weight.ndim)))
        .add(eps)
        .rsqrt()
        .view(return_shape)
    )


def weight_denorm(
    module: nn.Module,
    modulation_features: int,
    name: str = "weight",
    pre_normalize: str = "stylegan2",
    eps: float = 1e-9,
) -> nn.Module:
    WeightDeNorm.apply(
        module=module,
        modulation_features=modulation_features,
        name=name,
        pre_normalize=pre_normalize,
        eps=eps,
    )
    return module


def remove_weight_denorm(module: nn.Module) -> nn.Module:
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightDeNorm):
            hook.remove(module)  # WeightDeNorm removes WeightDeNormOutput too.
            del module._forward_pre_hooks[k]
            break
    return module
