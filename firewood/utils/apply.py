import torch.nn as nn

from firewood.common.collector import Collector
from firewood.layers.activations import BiasedActivation
from firewood.layers.denormalizations import AdaptiveNorm
from firewood.layers.normalizations import InstanceNorm


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
    for instance in Collector.layers():
        if isinstance(instance, (InstanceNorm, AdaptiveNorm)):
            instance.unbiased = value


def set_biased_activation_force_default(
    module: nn.Module, value: bool
) -> nn.Module:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    for submodule in module.modules():
        if isinstance(submodule, BiasedActivation):
            submodule.force_default = value
    return module


def set_all_biased_activation_force_default(value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    for instance in Collector.layers():
        if isinstance(instance, BiasedActivation):
            instance.force_default = value
