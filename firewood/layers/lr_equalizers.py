import math
from collections import OrderedDict
from typing import List, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn import Parameter

from firewood import utils

_NEED_RECURSIVE = {
    "AdaptiveNorm",
    "DepthSepConv1d",
    "DepthSepConv2d",
    "DepthSepConv3d",
    "DepthSepConvTranspose1d",
    "DepthSepConvTranspose2d",
    "DepthSepConvTranspose3d",
    "SpatialSepConv2d",
    "SpatialSepConv3d",
    "SpatialSepConvTranspose2d",
    "SpatialSepConvTranspose3d",
}


class BiasLREqualizer:
    def __init__(self, name: str = "bias") -> None:
        self.name = name
        self.target_name = self.name

    @staticmethod
    def apply(
        module: nn.Module,
        name: str = "bias",
        lr_multiplier: float = 1.0,
        init: Optional[Union[float, Tensor]] = None,
        recursive: bool = False,
        reapply: bool = False,
    ) -> Optional["BiasLREqualizer"]:
        if recursive:
            for _module in module.modules():
                BiasLREqualizer.apply(
                    module=_module,
                    name=name,
                    lr_multiplier=lr_multiplier,
                    init=init,
                    recursive=False,
                    reapply=reapply,
                )
            return None
        module_name = utils.get_name(module)
        if module_name not in _NEED_RECURSIVE and module_name.endswith("Norm"):
            return None
        if hasattr(module, "lr_equalization"):
            setattr(module, "lr_equalization", True)
        if getattr(module, name, None) is None:
            return None
        if init is None:
            init = 0.0
        if has_bias_lr_equalizer(module):
            if not reapply or getattr(module, "bias_init", None) == init:
                return None
            _remove_bias_lr_equalizer(module, recursive=False)

        fn = BiasLREqualizer(name=name)
        module.register_forward_pre_hook(fn)
        forward_pre_hooks = list(module._forward_pre_hooks.items())
        forward_pre_hooks = forward_pre_hooks[-1:] + forward_pre_hooks[:-1]
        module._forward_pre_hooks = OrderedDict(forward_pre_hooks)

        # other norm use `name + '_orig'` to save the original bias
        if hasattr(module, name + "_orig"):
            setattr(fn, "target_name", name + "_orig")

        bias: Tensor = utils.popattr(module, fn.target_name).clone()
        bias = torch.tensor(init, dtype=bias.dtype, device=bias.device).expand(
            bias.shape
        )
        setattr(module, "bias_init", init)
        setattr(module, "bias_gain", lr_multiplier)
        setattr(module, fn.target_name, bias.detach())
        module.register_parameter(name + "_param", Parameter(bias.clone()))
        return fn

    def remove(self, module: nn.Module) -> None:
        if hasattr(module, "lr_equalization"):
            setattr(module, "lr_equalization", False)
        with torch.no_grad():
            bias = self.compute_bias(module)
        delattr(module, self.name + "_param")
        delattr(module, "bias_init")
        delattr(module, "bias_gain")
        if hasattr(module, self.name + "_orig"):
            module.register_parameter(
                self.target_name, Parameter(bias.detach())
            )
        else:
            delattr(module, self.name)
            module.register_parameter(self.name, Parameter(bias.detach()))

    def compute_bias(self, module: nn.Module) -> Tensor:
        bias: Parameter = getattr(module, self.name + "_param")
        bias_gain = getattr(module, "bias_gain")
        if bias_gain != 1.0:
            bias = bias * bias_gain
        return bias

    def __call__(self, module: nn.Module, input: Tensor) -> None:
        setattr(module, self.target_name, self.compute_bias(module))


class WeightLREqualizer:
    """
    Note:
        LREqualizer hook should be applied after other weight norm hooks.
    """

    def __init__(self, name: str = "weight") -> None:
        self.name = name
        self.target_name = self.name

    @staticmethod
    def apply(
        module: nn.Module,
        name: str = "weight",
        lr_multiplier: float = 1.0,
        init_std: Optional[float] = None,
        recursive: bool = False,
        reapply: bool = False,
    ) -> Optional["WeightLREqualizer"]:
        if recursive:
            for _module in module.modules():
                WeightLREqualizer.apply(
                    module=_module,
                    name=name,
                    lr_multiplier=lr_multiplier,
                    init_std=init_std,
                    recursive=False,
                    reapply=reapply,
                )
            return None
        module_name = utils.get_name(module)
        if module_name not in _NEED_RECURSIVE and module_name.endswith("Norm"):
            return None
        if hasattr(module, "lr_equalization"):
            setattr(module, "lr_equalization", True)
        _weight: Optional[Tensor] = getattr(module, name, None)
        if _weight is None or _weight.ndim == 1:
            return None
        if init_std is None:
            init_std = 1.0
        if has_weight_lr_equalizer(module):
            if (
                not reapply
                or getattr(module, "weight_init_std", None) == init_std
            ):
                return None
            _remove_weight_lr_equalizer(module, recursive=False)

        fn = WeightLREqualizer(name=name)
        module.register_forward_pre_hook(fn)
        forward_pre_hooks = list(module._forward_pre_hooks.items())
        forward_pre_hooks = forward_pre_hooks[-1:] + forward_pre_hooks[:-1]
        module._forward_pre_hooks = OrderedDict(forward_pre_hooks)

        # other weight norm use `name + '_orig'` to save the original weight
        if hasattr(module, name + "_orig"):
            setattr(fn, "target_name", name + "_orig")

        weight: Tensor = utils.popattr(module, fn.target_name).clone()
        setattr(module, "weight_init_std", init_std)
        init.normal_(weight, mean=0, std=init_std / lr_multiplier)
        setattr(module, fn.target_name, weight.detach())
        module.register_parameter(
            name + "_param", Parameter(weight.detach().clone())
        )
        fan_in = weight.detach()[0].numel()
        weight_gain = lr_multiplier / math.sqrt(fan_in)
        setattr(module, "weight_gain", weight_gain)
        return fn

    def remove(self, module: nn.Module) -> None:
        if hasattr(module, "lr_equalization"):
            setattr(module, "lr_equalization", False)
        with torch.no_grad():
            weight = self.compute_weight(module).clone()
        delattr(module, self.name + "_param")
        delattr(module, "weight_init_std")
        delattr(module, "weight_gain")
        if hasattr(module, self.name + "_orig"):
            module.register_parameter(
                self.target_name, Parameter(weight.detach())
            )
        else:
            delattr(module, self.name)
            module.register_parameter(self.name, Parameter(weight.detach()))

    def compute_weight(self, module: nn.Module) -> Tensor:
        weight: Parameter = getattr(module, self.name + "_param")
        weight_gain = getattr(module, "weight_gain")
        if weight_gain != 1.0:
            weight = weight * weight_gain
        return weight

    def __call__(self, module: nn.Module, input: Tensor) -> None:
        # For the case of applying spectral norm after applying lr equalizer.
        if (
            self.target_name == self.name
            and getattr(module, self.name + "_orig", None) is not None
        ):
            self.target_name = self.name + "_orig"
        setattr(module, self.target_name, self.compute_weight(module))


def lr_equalizer(
    module: Union[
        nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]
    ],
    weight_name: str = "weight",
    bias_name: str = "bias",
    lr_multiplier: float = 1.0,
    weight_init_std: float = 1.0,
    bias_init: Optional[float] = None,
    recursive: bool = False,
    reapply: bool = False,
) -> Union[nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]]:
    if isinstance(module, (nn.ModuleList, list, tuple)):
        for _module in module:
            lr_equalizer(
                module=_module,
                weight_name=weight_name,
                bias_name=bias_name,
                lr_multiplier=lr_multiplier,
                weight_init_std=weight_init_std,
                bias_init=bias_init,
                recursive=recursive,
                reapply=reapply,
            )
        return module
    if (
        getattr(module, "weight_layer", None) is not None
        or utils.get_name(module) in _NEED_RECURSIVE
    ):
        recursive = True
    BiasLREqualizer.apply(
        module=module,
        name=bias_name,
        lr_multiplier=lr_multiplier,
        init=bias_init,
        recursive=recursive,
        reapply=reapply,
    )
    WeightLREqualizer.apply(
        module=module,
        name=weight_name,
        lr_multiplier=lr_multiplier,
        init_std=weight_init_std,
        recursive=recursive,
        reapply=reapply,
    )
    return module


def _remove_bias_lr_equalizer(
    module: nn.Module,
    recursive: bool = False,
) -> nn.Module:
    if recursive:
        for _module in module.modules():
            _remove_bias_lr_equalizer(_module, recursive=False)
        return module
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BiasLREqualizer):
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    return module


def _remove_weight_lr_equalizer(
    module: nn.Module,
    recursive: bool = False,
) -> nn.Module:
    if recursive:
        for _module in module.modules():
            _remove_weight_lr_equalizer(_module, recursive=False)
        return module
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightLREqualizer):
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    return module


def remove_lr_equalizer(
    module: Union[
        nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]
    ],
    recursive: bool = False,
) -> Union[nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]]:
    if isinstance(module, (nn.ModuleList, list, tuple)):
        for _module in module:
            remove_lr_equalizer(_module, recursive=recursive)
        return module
    if recursive:
        for _module in module.modules():
            remove_lr_equalizer(_module, recursive=False)
        return module
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, (WeightLREqualizer, BiasLREqualizer)):
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    return module


def has_bias_lr_equalizer(module: nn.Module) -> bool:
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, BiasLREqualizer):
            return True
    return False


def has_weight_lr_equalizer(module: nn.Module) -> bool:
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightLREqualizer):
            return True
    return False


def pop_bias_lr_equalizer(module: nn.Module) -> BiasLREqualizer:
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BiasLREqualizer):
            del module._forward_pre_hooks[k]
            return hook
    raise ValueError("No BiasLREqualizer found in module's forward pre hooks")


def pop_weight_lr_equalizer(module: nn.Module) -> WeightLREqualizer:
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightLREqualizer):
            del module._forward_pre_hooks[k]
            return hook
    raise ValueError("No WeightLREqualizer found in module's forward pre hooks")


class BIAS_ATTRS(TypedDict):
    bias: Parameter
    bias_init: Optional[Union[float, Tensor]]
    bias_gain: float
    bias_hook: Optional[BiasLREqualizer]


def pop_bias_attrs(
    module: nn.Module,
) -> BIAS_ATTRS:
    name = "bias"

    bias_init = getattr(module, "bias_init", None)
    bias_gain = getattr(module, "bias_gain", 1.0)

    bias_hook = None
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BiasLREqualizer):
            hook.remove(module)
            del module._forward_pre_hooks[k]
            name = hook.name
            bias_hook = hook
            break
    if bias_hook is not None:
        bias_hook.name = "bias"

    if not hasattr(module, name):
        raise ValueError(f"No bias found in module {module}")
    bias: Parameter = utils.popattr(module, name)
    module.register_parameter(name, None)

    if hasattr(module, "bias_init"):
        delattr(module, "bias_init")

    if hasattr(module, "bias_gain"):
        delattr(module, "bias_gain")

    return {
        "bias": bias,
        "bias_init": bias_init,
        "bias_gain": bias_gain,
        "bias_hook": bias_hook,
    }


def set_bias_attrs(
    module: nn.Module,
    bias: Parameter,
    bias_init: Optional[Union[float, Tensor]] = None,
    bias_gain: float = 1.0,
    bias_hook: Optional[BiasLREqualizer] = None,
    reset_bias: bool = True,
) -> nn.Module:
    name = bias_hook.name if bias_hook is not None else "bias"
    utils.popattr(module, "bias", None)
    module.register_parameter(name, bias)
    if bias_hook is None:
        return module

    BiasLREqualizer.apply(
        module=module,
        name=name,
        lr_multiplier=bias_gain,
        init=bias_init,
        recursive=False,
    )
    if not reset_bias:
        delattr(module, name + "_param")
        module.register_parameter(name + "_param", Parameter(bias.detach()))
    return module


def transfer_bias_attrs(
    source_module: nn.Module,
    target_module: nn.Module,
    reset_bias: bool = True,
) -> nn.Module:
    return set_bias_attrs(
        module=target_module,
        reset_bias=reset_bias,
        **pop_bias_attrs(source_module),
    )
