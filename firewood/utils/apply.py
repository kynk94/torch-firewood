from typing import Any, Type

import torch.nn as nn

from firewood.common.collector import Collector
from firewood.hooks import weight_denormalizations
from firewood.layers import biased_activations, conv_gradfix
from firewood.layers.activations import normalize_activation_name
from firewood.layers.denormalizations import AdaptiveNorm
from firewood.layers.normalizations import InstanceNorm


def set_all_module_attr(cls: Type[nn.Module], attr: str, value: Any) -> None:
    if not isinstance(attr, str):
        raise TypeError(f"Attr should be a str, got {type(attr)}")
    for layer in Collector.layers():
        if isinstance(layer, cls):
            setattr(layer, attr, value)


def set_all_conv_force_default(value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    conv_gradfix.FORCE_DEFAULT = value
    for layer in Collector.layers():
        if isinstance(layer, conv_gradfix._GFixConvNd):
            layer.force_default = value


def set_all_weight_denorm_force_default(value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    weight_denormalizations.FORCE_DEFAULT = value
    for hook in Collector.hooks():
        if isinstance(hook, weight_denormalizations.WeightDeNorm):
            hook.force_default = value


def set_instance_norm_unbiased(module: nn.Module, value: bool) -> nn.Module:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    for submodule in module.modules():
        if isinstance(submodule, (InstanceNorm, AdaptiveNorm)):
            submodule.unbiased = value
    return module


def set_all_instance_norm_unbiased(value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    for layer in Collector.layers():
        if isinstance(layer, (InstanceNorm, AdaptiveNorm)):
            layer.unbiased = value


def set_biased_activation_force_default(
    module: nn.Module, value: bool
) -> nn.Module:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    for submodule in module.modules():
        if isinstance(submodule, biased_activations.BiasedActivation):
            submodule.force_default = value
    return module


def set_all_biased_activation_force_default(value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    biased_activations.FORCE_DEFAULT = value
    for layer in Collector.layers():
        if isinstance(layer, biased_activations.BiasedActivation):
            layer.force_default = value


def set_all_biased_activation_gain(activation: str, value: float) -> None:
    if not isinstance(activation, str):
        raise TypeError(f"Activation should be a str, got {type(activation)}")
    if not isinstance(value, float):
        raise TypeError(f"Expected float, got {type(value)}")
    activation = normalize_activation_name(activation)
    for layer in Collector.layers():
        if (
            isinstance(layer, biased_activations.BiasedActivation)
            and layer.activation == activation
        ):
            layer.gain = value
