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
