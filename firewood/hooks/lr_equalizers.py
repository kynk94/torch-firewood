import math
import re
from collections import OrderedDict
from typing import (
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
    overload,
)

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

    def __check_target_name_with_orig(self, module: nn.Module) -> None:
        if (
            hasattr(module, self.name + "_orig")
            and self.target_name == self.name
        ):
            self.target_name = self.name + "_orig"

    def register(self, module: nn.Module) -> None:
        module.register_forward_pre_hook(self)
        forward_pre_hooks = list(module._forward_pre_hooks.items())
        module._forward_pre_hooks = OrderedDict(
            [forward_pre_hooks.pop()] + forward_pre_hooks
        )
        self.__check_target_name_with_orig(module=module)

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
        _set_lr_equalization(module=module, value=False)

    def compute(self, module: nn.Module) -> Tensor:
        parameter: Tensor = getattr(module, self.name + "_param")
        gain: Tensor = getattr(module, f"_{self.name}_gain")
        if gain != 1.0:
            parameter = parameter * gain.to(parameter)
        return parameter

    def __call__(self, module: nn.Module, input: Tensor) -> None:
        self.__check_target_name_with_orig(module=module)
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
        recursive: bool = True,
        keep_value: bool = False,
    ) -> Optional["BiasLREqualizer"]:
        def __recursive_apply() -> Optional[BiasLREqualizer]:
            fn: Optional[BiasLREqualizer] = None
            for _module in module.modules():
                _fn = BiasLREqualizer.apply(
                    module=_module,
                    name=name,
                    lr_multiplier=lr_multiplier,
                    init=init,
                    recursive=False,
                    keep_value=keep_value,
                )
                if fn is None:
                    fn = _fn
            return fn

        if utils.attr_is_value(module, "lr_equalization", False):
            _set_lr_equalization(module=module, value=True)
            return __recursive_apply()
        if recursive:
            return __recursive_apply()
        if not BiasLREqualizer.is_applicable(module=module, name=name):
            return None

        fn = BiasLREqualizer(name=name)
        fn.register(module=module)
        fn.reset_parameters(module, lr_multiplier, init, keep_value)
        return fn

    @torch.no_grad()
    def reset_parameters(
        self,
        module: nn.Module,
        lr_multiplier: Optional[Union[float, Tensor]] = None,
        init: Optional[float] = None,
        keep_value: bool = False,
    ) -> None:
        gain_name = f"_{self.name}_gain"
        init_name = f"_{self.name}_init"
        if lr_multiplier is None:
            lr_multiplier = getattr(module, gain_name, 1.0)
        if not isinstance(lr_multiplier, Tensor):
            lr_multiplier = torch.tensor(lr_multiplier, dtype=torch.float32)
        if init is None:
            init = cast(float, getattr(module, init_name, 0.0))

        original_bias: Tensor = utils.popattr(module, self.target_name)
        if keep_value:
            bias = Parameter(original_bias / lr_multiplier.to(original_bias))
        else:
            bias = Parameter(torch.full_like(original_bias, fill_value=init))

        module.register_parameter(f"{self.name}_param", bias)
        setattr(module, self.target_name, bias.detach())
        module.register_buffer(gain_name, lr_multiplier)
        setattr(module, init_name, init)


class WeightLREqualizer(_LREqualizer):
    def __init__(self, name: str = "weight") -> None:
        super().__init__(name=name)

    @staticmethod
    def apply(
        module: nn.Module,
        name: str = "weight",
        lr_multiplier: float = 1.0,
        init_std: float = 1.0,
        recursive: bool = True,
        keep_value: bool = False,
    ) -> Optional["WeightLREqualizer"]:
        def __recursive_apply() -> Optional[WeightLREqualizer]:
            fn: Optional[WeightLREqualizer] = None
            for _module in module.modules():
                _fn = WeightLREqualizer.apply(
                    module=_module,
                    name=name,
                    lr_multiplier=lr_multiplier,
                    init_std=init_std,
                    recursive=False,
                    keep_value=keep_value,
                )
                if fn is None:
                    fn = _fn
            return fn

        if utils.attr_is_value(module, "lr_equalization", False):
            _set_lr_equalization(module=module, value=True)
            return __recursive_apply()
        if recursive:
            return __recursive_apply()
        if not WeightLREqualizer.is_applicable(module=module, name=name):
            return None

        fn = WeightLREqualizer(name=name)
        fn.register(module=module)
        fn.reset_parameters(module, lr_multiplier, init_std, keep_value)
        return fn

    @torch.no_grad()
    def reset_parameters(
        self,
        module: nn.Module,
        lr_multiplier: Optional[Union[float, Tensor]] = None,
        init_std: Optional[float] = None,
        keep_value: bool = False,
    ) -> None:
        gain_name = f"_{self.name}_gain"
        init_name = f"_{self.name}_init"
        if lr_multiplier is None:
            lr_multiplier = cast(float, getattr(module, gain_name, 1.0))
        if not isinstance(lr_multiplier, Tensor):
            lr_multiplier = torch.tensor(lr_multiplier, dtype=torch.float32)
        if init_std is None:
            init_std = cast(float, getattr(module, init_name, 1.0))

        original_weight: Tensor = utils.popattr(module, self.target_name)
        fan_in = original_weight.detach()[0].numel()
        gain = lr_multiplier / math.sqrt(fan_in)
        if keep_value:
            weight = Parameter(original_weight / gain.to(original_weight))
        else:
            weight = Parameter(original_weight.detach())
            init.normal_(weight, mean=0, std=init_std / lr_multiplier)

        module.register_parameter(f"{self.name}_param", weight)
        setattr(module, self.target_name, weight.detach())
        module.register_buffer(gain_name, gain)
        setattr(module, init_name, init_std)


def _set_lr_equalization(module: nn.Module, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError("lr_equalization must be bool type.")
    if utils.attr_is_value(module, "lr_equalization", not value):
        setattr(module, "lr_equalization", value)


def lr_equalizer(
    module: Union[
        nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]
    ],
    weight_name: str = "weight",
    bias_name: str = "bias",
    lr_multiplier: float = 1.0,
    weight_init_std: float = 1.0,
    bias_init: float = 0.0,
    keep_value: bool = False,
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
                keep_value=keep_value,
            )
        return module
    BiasLREqualizer.apply(
        module=module,
        name=bias_name,
        lr_multiplier=lr_multiplier,
        init=bias_init,
        recursive=True,
        keep_value=keep_value,
    )
    WeightLREqualizer.apply(
        module=module,
        name=weight_name,
        lr_multiplier=lr_multiplier,
        init_std=weight_init_std,
        recursive=True,
        keep_value=keep_value,
    )
    return module


def remove_lr_equalizer(
    module: Union[
        nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]
    ]
) -> Union[nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]]:
    if isinstance(module, (nn.ModuleList, list, tuple)):
        for _module in module:
            if _module is None:
                continue
            remove_lr_equalizer(_module)
        return module
    for _module in module.modules():
        _pop_bias_lr_equalizer(_module, raise_exception=False)
    for _module in module.modules():
        _pop_weight_lr_equalizer(_module, raise_exception=False)
    return module


@overload
def _pop_bias_lr_equalizer(
    module: nn.Module, raise_exception: Literal[True]
) -> BiasLREqualizer:
    ...


@overload
def _pop_bias_lr_equalizer(
    module: nn.Module, raise_exception: Literal[False]
) -> Optional[BiasLREqualizer]:
    ...


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


@overload
def _pop_weight_lr_equalizer(
    module: nn.Module, raise_exception: Literal[True]
) -> WeightLREqualizer:
    ...


@overload
def _pop_weight_lr_equalizer(
    module: nn.Module, raise_exception: Literal[False]
) -> Optional[WeightLREqualizer]:
    ...


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
        bias_hook = _pop_bias_lr_equalizer(module, raise_exception=True)
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
