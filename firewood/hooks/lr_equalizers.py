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
from firewood.hooks.hook import _Hook

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
_EXCLUDE_PATTERN = {
    r"Norm(\dd)?$",
    r"Noise$",
}


class _LREqualizer(_Hook):
    def __init__(self, name: str) -> None:
        if name.endswith("_orig"):
            self.call_name = name
            self.name = name.rstrip("_orig")
        else:
            self.call_name = self.name = name

    @staticmethod
    def is_applicable(module: nn.Module, name: str) -> bool:
        module_name = utils.get_name(module)
        if module_name not in _NEED_RECURSIVE and any(
            re.search(pattern, module_name) for pattern in _EXCLUDE_PATTERN
        ):
            return False
        parameter = getattr(module, name, None)
        if parameter is None:
            return False
        return True

    @staticmethod
    def is_registered(module: nn.Module, name: str) -> bool:
        return hasattr(module, name + "_param")

    def __check_call_name_with_orig(self, module: nn.Module) -> None:
        if hasattr(module, self.name + "_orig") and self.call_name == self.name:
            self.call_name = self.name + "_orig"

    def register(self, module: nn.Module) -> None:
        module.register_forward_pre_hook(self)
        forward_pre_hooks = list(module._forward_pre_hooks.items())
        module._forward_pre_hooks = OrderedDict(
            [forward_pre_hooks.pop()] + forward_pre_hooks
        )
        self.__check_call_name_with_orig(module=module)

    def remove(self, module: nn.Module) -> None:
        with torch.no_grad():
            parameter = self.compute(module)
        parameter = Parameter(parameter.detach())

        delattr(module, self.name + "_param")
        delattr(module, f"_{self.name}_coeff")
        delattr(module, f"_{self.name}_init")
        if hasattr(module, self.call_name):
            delattr(module, self.call_name)
            module.register_parameter(self.call_name, parameter)
        elif isinstance(getattr(module, self.name, None), Parameter):
            delattr(module, self.name)
            module.register_parameter(self.name, parameter)

    def compute(self, module: nn.Module) -> Tensor:
        parameter: Tensor = getattr(module, self.name + "_param")
        coeff: Tensor = getattr(module, f"_{self.name}_coeff")
        return parameter * coeff.to(dtype=parameter.dtype)

    def __call__(self, module: nn.Module, input: Tensor) -> None:
        self.__check_call_name_with_orig(module=module)
        setattr(module, self.call_name, self.compute(module))


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
        def __recursive_apply() -> BiasLREqualizer:
            fn: Optional[BiasLREqualizer] = None
            for submodule in module.modules():
                _fn = BiasLREqualizer.apply(
                    module=submodule,
                    name=name,
                    lr_multiplier=lr_multiplier,
                    init=init,
                    recursive=False,
                    keep_value=keep_value,
                )
                if fn is None:
                    fn = _fn
            return cast(BiasLREqualizer, fn)

        if recursive:
            return __recursive_apply()
        if not _LREqualizer.is_applicable(module=module, name=name):
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
        coeff_name = f"_{self.name}_coeff"
        init_name = f"_{self.name}_init"

        original_bias: Tensor = utils.popattr(module, self.call_name)
        original_coeff: Optional[Tensor] = getattr(module, coeff_name, None)
        if original_coeff is not None:
            original_bias = original_bias * original_coeff

        if lr_multiplier is None:
            lr_multiplier = original_coeff or 1.0
        if not isinstance(lr_multiplier, Tensor):
            lr_multiplier = torch.tensor(lr_multiplier, dtype=torch.float32)
        if init is None:
            init = cast(float, getattr(module, init_name, 0.0))

        if keep_value:
            bias = Parameter(original_bias / lr_multiplier.to(original_bias))
        else:
            bias = Parameter(torch.full_like(original_bias, fill_value=init))

        module.register_parameter(f"{self.name}_param", bias)
        setattr(module, self.call_name, bias.detach())
        module.register_buffer(coeff_name, lr_multiplier)
        setattr(module, init_name, init)


class WeightLREqualizer(_LREqualizer):
    def __init__(self, name: str = "weight") -> None:
        super().__init__(name=name)

    @staticmethod
    def apply(
        module: nn.Module,
        name: str = "weight",
        lr_multiplier: float = 1.0,
        gain: Optional[float] = None,
        init_std: float = 1.0,
        recursive: bool = True,
        keep_value: bool = False,
    ) -> Optional["WeightLREqualizer"]:
        def __recursive_apply() -> WeightLREqualizer:
            fn: Optional[WeightLREqualizer] = None
            for submodule in module.modules():
                _fn = WeightLREqualizer.apply(
                    module=submodule,
                    name=name,
                    lr_multiplier=lr_multiplier,
                    gain=gain,
                    init_std=init_std,
                    recursive=False,
                    keep_value=keep_value,
                )
                if fn is None:
                    fn = _fn
            return cast(WeightLREqualizer, fn)

        if recursive:
            return __recursive_apply()
        if not _LREqualizer.is_applicable(module=module, name=name):
            return None

        fn = WeightLREqualizer(name=name)
        fn.register(module=module)
        fn.reset_parameters(module, lr_multiplier, gain, init_std, keep_value)
        return fn

    @torch.no_grad()
    def reset_parameters(
        self,
        module: nn.Module,
        lr_multiplier: Optional[Union[float, Tensor]] = None,
        gain: Optional[Union[float, Tensor]] = None,
        init_std: Optional[float] = None,
        keep_value: bool = False,
    ) -> None:
        # gain_name does not start with "_", and does not setattr while applying
        # Only manually set gain_name will be used.
        gain_name = f"{self.name}_gain"
        coeff_name = f"_{self.name}_coeff"
        init_name = f"_{self.name}_init"

        original_weight: Tensor = utils.popattr(module, self.call_name)
        original_gain: Optional[Tensor] = getattr(module, gain_name, None)
        original_coeff: Optional[Tensor] = getattr(module, coeff_name, None)
        if original_coeff is not None:
            original_weight = original_weight * original_coeff

        fan_in = original_weight.detach()[0].numel()
        if lr_multiplier is None:
            lr_multiplier = 1.0
        if not isinstance(lr_multiplier, Tensor):
            lr_multiplier = torch.tensor(lr_multiplier, dtype=torch.float32)
        if original_coeff is not None:
            original_coeff *= math.sqrt(fan_in) / lr_multiplier
        if gain is None:
            gain = original_gain or 1.0
        if init_std is None:
            init_std = cast(float, getattr(module, init_name, 1.0))
        coeff = gain * lr_multiplier / math.sqrt(fan_in)

        if keep_value:
            weight = Parameter(original_weight / coeff.to(original_weight))
        else:
            weight = Parameter(original_weight.detach())
            init.normal_(weight, mean=0, std=init_std / lr_multiplier)

        module.register_parameter(f"{self.name}_param", weight)
        setattr(module, self.call_name, weight.detach())
        module.register_buffer(coeff_name, coeff)
        setattr(module, init_name, init_std)


def lr_equalizer(
    module: Union[
        nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]
    ],
    weight_name: str = "weight",
    bias_name: str = "bias",
    lr_multiplier: float = 1.0,
    weight_gain: Optional[float] = None,
    weight_init_std: float = 1.0,
    bias_init: float = 0.0,
    keep_value: bool = False,
) -> Union[nn.Module, nn.ModuleList, List[nn.Module], Tuple[nn.Module, ...]]:
    if isinstance(module, (nn.ModuleList, list, tuple)):
        for submodule in module:
            if submodule is None:
                continue
            lr_equalizer(
                module=submodule,
                weight_name=weight_name,
                bias_name=bias_name,
                lr_multiplier=lr_multiplier,
                weight_gain=weight_gain,
                weight_init_std=weight_init_std,
                bias_init=bias_init,
                keep_value=keep_value,
            )
        return module
    WeightLREqualizer.apply(
        module=module,
        name=weight_name,
        lr_multiplier=lr_multiplier,
        gain=weight_gain,
        init_std=weight_init_std,
        recursive=True,
        keep_value=keep_value,
    )
    BiasLREqualizer.apply(
        module=module,
        name=bias_name,
        lr_multiplier=lr_multiplier,
        init=bias_init,
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
        for submodule in module:
            if submodule is None:
                continue
            remove_lr_equalizer(submodule)
        return module
    for submodule in module.modules():
        _pop_weight_lr_equalizer(submodule, raise_exception=False)
    for submodule in module.modules():
        _pop_bias_lr_equalizer(submodule, raise_exception=False)
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
    coeff: float
    init: float
    use_hook: bool


def pop_bias_attrs(
    module: nn.Module,
) -> BIAS_ATTRS:
    bias_coeff = getattr(module, "_bias_coeff", 1.0)
    bias_init = getattr(module, "_bias_init", 0.0)
    try:
        hook = _pop_bias_lr_equalizer(module, raise_exception=True)
        original_name = hook.name
        use_hook = True
    except RuntimeError:
        original_name = "bias"
        use_hook = False

    for name, param in module.named_parameters(recurse=False):
        if "bias" in name or name == original_name:
            bias = cast(Parameter, utils.popattr(module, name))
            module.register_parameter(name, None)
            break
    else:
        raise ValueError("Bias is not found in the module.")
    return {
        "bias": bias,
        "coeff": bias_coeff,
        "init": bias_init,
        "use_hook": use_hook,
    }


def set_bias_attrs(
    module: nn.Module,
    bias: Parameter,
    coeff: float = 1.0,
    init: float = 0.0,
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
        lr_multiplier=coeff,
        init=init,
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
