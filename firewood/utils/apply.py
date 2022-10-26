import torch.nn as nn


def set_instance_norm_unbiased(module: nn.Module, value: bool) -> nn.Module:
    # prevent circular import
    from firewood.layers.denormalizations import AdaptiveNorm
    from firewood.layers.normalizations import InstanceNorm

    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    for submodule in module.modules():
        if isinstance(submodule, (InstanceNorm, AdaptiveNorm)):
            submodule.unbiased = value
    return module


def set_biased_activation_force_default(
    module: nn.Module, value: bool
) -> nn.Module:
    # prevent circular import
    from firewood.layers.activations import BiasedActivation

    if not isinstance(value, bool):
        raise TypeError(f"Expected bool, got {type(value)}")
    for submodule in module.modules():
        if isinstance(submodule, BiasedActivation):
            submodule.force_default = value
    return module
