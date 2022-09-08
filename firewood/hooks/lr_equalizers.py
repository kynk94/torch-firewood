import math
import re
from typing import List, Optional, Tuple, TypedDict, Union, cast

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


class _LREqualizer:
    def __init__(self, name: str) -> None:
        self.target_name = self.name = name

    @staticmethod
    def set_lr_equalization(module: nn.Module, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("lr_equalization must be bool type.")
        if utils.attr_is_value(module, "lr_equalization", not value):
            setattr(module, "lr_equalization", value)

    @staticmethod
    def is_applicable(module: nn.Module, name: str) -> bool:
        module_name = utils.get_name(module)
        if module_name not in _NEED_RECURSIVE and re.search(
            r"Norm(\dd)?$", module_name
        ):
            return False
        parameter = getattr(module, name, None)
        if parameter is None:
            return False
        if not isinstance(parameter, Parameter):
            return False
        return True

    def register(self, module: nn.Module) -> None:
        utils.register_forward_pre_hook_to_index(
            module=module, hook=self, index=0
        )
        self.has_orig(module=module)

    def is_registered(self, module: nn.Module) -> bool:
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, type(self)):
                return True
        return False

    def has_orig(self, module: nn.Module) -> None:
        if (
            hasattr(module, self.name + "_orig")
            and self.target_name == self.name
        ):
            self.target_name = self.name + "_orig"

    def remove(self, module: nn.Module) -> None:
        with torch.no_grad():
            parameter = self.compute(module)
        delattr(module, self.name + "_param")
        delattr(module, f"_{self.name}_gain")
        delattr(module, f"_{self.name}_init")
        delattr(module, self.target_name)
        module.register_parameter(
            self.target_name, Parameter(parameter.detach())
        )
        self.set_lr_equalization(module=module, value=False)

    def compute(self, module: nn.Module) -> Tensor:
        parameter: Parameter = getattr(module, self.name + "_param")
        gain = getattr(module, f"_{self.name}_gain")
        if gain != 1.0:
            parameter = parameter * gain
        return parameter

    def __call__(self, module: nn.Module, input: Tensor) -> None:
        self.has_orig(module=module)
        setattr(module, self.target_name, self.compute(module))


class BiasLREqualizer(_LREqualizer):
    def __init__(self, name: str = "bias") -> None:
        super().__init__(name=name)

    @staticmethod
    def apply(
        module: nn.Module,
        name: str = "bias",
        lr_multiplier: float = 1.0,
        init: float = 0.0,
        recursive: bool = False,
    ) -> Optional["BiasLREqualizer"]:
        def __recursive_apply() -> None:
            for _module in module.modules():
                BiasLREqualizer.apply(
                    module=_module,
                    name=name,
                    lr_multiplier=lr_multiplier,
                    init=init,
                    recursive=False,
                )

        if utils.attr_is_value(module, "lr_equalization", False):
            BiasLREqualizer.set_lr_equalization(module=module, value=True)
            __recursive_apply()
            return None
        if recursive:
            __recursive_apply()
            return None
        if not BiasLREqualizer.is_applicable(module=module, name=name):
            return None

        BiasLREqualizer.set_lr_equalization(module=module, value=True)
        fn = BiasLREqualizer(name=name)
        fn.register(module=module)
        fn.reset_parameters(module, lr_multiplier, init)
        return fn

    def reset_parameters(
        self,
        module: nn.Module,
        lr_multiplier: Optional[float] = None,
        init: Optional[float] = None,
    ) -> None:
        gain_name = f"_{self.name}_gain"
        init_name = f"_{self.name}_init"
        if lr_multiplier is None:
            lr_multiplier = getattr(module, gain_name, 1.0)
        if init is None:
            init = cast(float, getattr(module, init_name, 0.0))

        original_bias: Tensor = utils.popattr(module, self.target_name)
        bias = Parameter(
            torch.full(
                size=original_bias.shape,
                fill_value=init,
                dtype=original_bias.dtype,
                device=original_bias.device,
            )
        )

        module.register_parameter(f"{self.name}_param", bias)
        setattr(module, self.target_name, bias.detach())
        setattr(module, gain_name, lr_multiplier)
        setattr(module, init_name, init)


class WeightLREqualizer(_LREqualizer):
    """
    Note:
        LREqualizer hook should be applied after other weight norm hooks.
    """

    def __init__(self, name: str = "weight") -> None:
        super().__init__(name=name)

    @staticmethod
    def apply(
        module: nn.Module,
        name: str = "weight",
        lr_multiplier: float = 1.0,
        init_std: float = 1.0,
        recursive: bool = False,
    ) -> Optional["WeightLREqualizer"]:
        def __recursive_apply() -> None:
            for _module in module.modules():
                WeightLREqualizer.apply(
                    module=_module,
                    name=name,
                    lr_multiplier=lr_multiplier,
                    init_std=init_std,
                    recursive=False,
                )

        if utils.attr_is_value(module, "lr_equalization", False):
            WeightLREqualizer.set_lr_equalization(module=module, value=True)
            __recursive_apply()
            return None
        if recursive:
            __recursive_apply()
            return None
        if not WeightLREqualizer.is_applicable(module=module, name=name):
            return None

        fn = WeightLREqualizer(name=name)
        fn.register(module=module)
        fn.reset_parameters(module, lr_multiplier, init_std)
        return fn

    def reset_parameters(
        self,
        module: nn.Module,
        lr_multiplier: Optional[float] = None,
        init_std: Optional[float] = None,
    ) -> None:
        gain_name = f"_{self.name}_gain"
        init_name = f"_{self.name}_init"
        if lr_multiplier is None:
            lr_multiplier = cast(float, getattr(module, gain_name, 1.0))
        if init_std is None:
            init_std = cast(float, getattr(module, init_name, 1.0))

        original_weight: Tensor = utils.popattr(module, self.target_name)
        weight = Parameter(original_weight.detach())
        init.normal_(weight, mean=0, std=init_std / lr_multiplier)
        fan_in = weight.detach()[0].numel()
        weight_gain = lr_multiplier / math.sqrt(fan_in)

        module.register_parameter(f"{self.name}_param", weight)
        setattr(module, self.target_name, weight.detach())
        setattr(module, gain_name, weight_gain)
        setattr(module, init_name, init_std)


def lr_equalizer(
    module: Union[
        nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]
    ],
    weight_name: str = "weight",
    bias_name: str = "bias",
    lr_multiplier: float = 1.0,
    weight_init_std: float = 1.0,
    bias_init: float = 0.0,
    recursive: bool = False,
) -> Union[nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]]:
    if isinstance(module, (nn.ModuleList, list, tuple)):
        for _module in module:
            if _module is None:
                continue
            lr_equalizer(
                module=_module,
                weight_name=weight_name,
                bias_name=bias_name,
                lr_multiplier=lr_multiplier,
                weight_init_std=weight_init_std,
                bias_init=bias_init,
                recursive=recursive,
            )
        return module
    if utils.get_name(module) in _NEED_RECURSIVE:
        recursive = True
    BiasLREqualizer.apply(
        module=module,
        name=bias_name,
        lr_multiplier=lr_multiplier,
        init=bias_init,
        recursive=recursive,
    )
    WeightLREqualizer.apply(
        module=module,
        name=weight_name,
        lr_multiplier=lr_multiplier,
        init_std=weight_init_std,
        recursive=recursive,
    )
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
    _remove_bias_lr_equalizer(module, recursive=False)
    _remove_weight_lr_equalizer(module, recursive=False)
    return module


def _pop_bias_lr_equalizer(
    module: nn.Module, raise_exception: bool = False
) -> Optional[BiasLREqualizer]:
    if utils.attr_is_value(module, "lr_equalization", True):
        setattr(module, "lr_equalization", False)
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BiasLREqualizer):
            hook.remove(module)
            return module._forward_pre_hooks.pop(k)
    if raise_exception:
        raise RuntimeError(
            "BiasLREqualizer is not found in module's forward-pre-hooks."
        )


def _pop_weight_lr_equalizer(
    module: nn.Module, raise_exception: bool = False
) -> Optional[WeightLREqualizer]:
    if utils.attr_is_value(module, "lr_equalization", True):
        setattr(module, "lr_equalization", False)
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightLREqualizer):
            hook.remove(module)
            return module._forward_pre_hooks.pop(k)
    if raise_exception:
        raise RuntimeError(
            "WeightLREqualizer is not found in module's forward-pre-hooks."
        )


def _remove_bias_lr_equalizer(
    module: nn.Module,
    recursive: bool = False,
) -> nn.Module:
    if recursive:
        for _module in module.modules():
            _remove_bias_lr_equalizer(_module, recursive=False)
        return module
    _pop_bias_lr_equalizer(module, raise_exception=False)
    return module


def _remove_weight_lr_equalizer(
    module: nn.Module,
    recursive: bool = False,
) -> nn.Module:
    if recursive:
        for _module in module.modules():
            _remove_weight_lr_equalizer(_module, recursive=False)
        return module
    _pop_weight_lr_equalizer(module, raise_exception=False)
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


class BIAS_ATTRS(TypedDict):
    bias: Parameter
    bias_gain: float
    bias_init: float
    use_hook: bool


def pop_bias_attrs(
    module: nn.Module,
) -> BIAS_ATTRS:
    bias_gain = getattr(module, "_bias_gain", 1.0)
    bias_init = getattr(module, "_bias_init", 0.0)
    try:
        bias_hook = cast(
            BiasLREqualizer,
            _pop_bias_lr_equalizer(module, raise_exception=True),
        )
        name, bias_hook.name = bias_hook.name, "bias"
        use_hook = True
    except RuntimeError:
        name = "bias"
        use_hook = False
    try:
        bias: Parameter = utils.popattr(module, name)
        module.register_parameter(name, None)
    except AttributeError:
        raise
    return {
        "bias": bias,
        "bias_gain": bias_gain,
        "bias_init": bias_init,
        "use_hook": use_hook,
    }


def set_bias_attrs(
    module: nn.Module,
    bias: Parameter,
    bias_gain: float = 1.0,
    bias_init: float = 0.0,
    use_hook: bool = False,
    reset_bias: bool = True,
) -> nn.Module:
    name = "bias"
    utils.popattr(module, name, None)
    module.register_parameter(name, bias)
    if not use_hook:
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
