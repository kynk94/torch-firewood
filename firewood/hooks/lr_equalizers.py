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

    def register(self, module: nn.Module) -> None:
        utils.register_forward_pre_hook_to_index(
            module=module, hook=self, index=0
        )


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
        if utils.attr_is_value(module, "lr_equalization", False):
            setattr(module, "lr_equalization", True)
        module_name = utils.get_name(module)
        if module_name not in _NEED_RECURSIVE and re.search(
            r"Norm(\dd)?$", module_name
        ):
            return None
        _bias = getattr(module, name, None)
        if _bias is None or not isinstance(_bias, Tensor):
            return None
        if has_bias_lr_equalizer(module):
            if not reapply or getattr(module, "_bias_init", None) == init:
                return None
            _remove_bias_lr_equalizer(module, recursive=False)

        fn = BiasLREqualizer(name=name)
        fn.register(module=module)

        # other norm use `name + '_orig'` to save the original bias
        if hasattr(module, name + "_orig"):
            setattr(fn, "target_name", name + "_orig")

        original_bias: Tensor = utils.popattr(module, fn.target_name)
        bias = Parameter(
            torch.full(
                size=original_bias.shape,
                fill_value=init,
                dtype=original_bias.dtype,
                device=original_bias.device,
            )
        )
        module.register_parameter(name + "_param", bias)
        setattr(module, fn.target_name, bias.detach())
        setattr(module, "_bias_gain", lr_multiplier)
        setattr(module, "_bias_init", init)
        return fn

    def remove(self, module: nn.Module) -> None:
        if utils.attr_is_value(module, "lr_equalization", True):
            setattr(module, "lr_equalization", False)
        with torch.no_grad():
            bias = self.compute_bias(module)
        delattr(module, self.name + "_param")
        delattr(module, "_bias_gain")
        delattr(module, "_bias_init")
        if hasattr(module, self.name + "_orig"):
            module.register_parameter(
                self.target_name, Parameter(bias.detach())
            )
        else:
            delattr(module, self.name)
            module.register_parameter(self.name, Parameter(bias.detach()))

    def compute_bias(self, module: nn.Module) -> Tensor:
        bias: Tensor = getattr(module, self.name + "_param")
        bias_gain: float = getattr(module, "_bias_gain")
        if bias_gain != 1.0:
            bias = bias * bias_gain
        return bias

    def __call__(self, module: nn.Module, input: Tensor) -> None:
        setattr(module, self.target_name, self.compute_bias(module))


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
        if utils.attr_is_value(module, "lr_equalization", False):
            setattr(module, "lr_equalization", True)
        module_name = utils.get_name(module)
        if module_name not in _NEED_RECURSIVE and re.search(
            r"Norm(\dd)?$", module_name
        ):
            return None
        _weight: Optional[Tensor] = getattr(module, name, None)
        if _weight is None or _weight.ndim == 1:
            return None
        if has_weight_lr_equalizer(module):
            if (
                not reapply
                or getattr(module, "_weight_init_std", None) == init_std
            ):
                return None
            _remove_weight_lr_equalizer(module, recursive=False)

        fn = WeightLREqualizer(name=name)
        fn.register(module=module)

        # other weight norm use `name + '_orig'` to save the original weight
        if hasattr(module, name + "_orig"):
            setattr(fn, "target_name", name + "_orig")

        weight: Tensor = utils.popattr(module, fn.target_name).clone()
        setattr(module, "_weight_init_std", init_std)
        init.normal_(weight, mean=0, std=init_std / lr_multiplier)
        setattr(module, fn.target_name, weight.detach())
        module.register_parameter(
            name + "_param", Parameter(weight.detach().clone())
        )
        fan_in = weight.detach()[0].numel()
        weight_gain = lr_multiplier / math.sqrt(fan_in)
        setattr(module, "_weight_gain", weight_gain)
        return fn

    def remove(self, module: nn.Module) -> None:
        if utils.attr_is_value(module, "lr_equalization", True):
            setattr(module, "lr_equalization", False)
        with torch.no_grad():
            weight = self.compute_weight(module).clone()
        delattr(module, self.name + "_param")
        delattr(module, "_weight_init_std")
        delattr(module, "_weight_gain")
        if hasattr(module, self.name + "_orig"):
            module.register_parameter(
                self.target_name, Parameter(weight.detach())
            )
        else:
            delattr(module, self.name)
            module.register_parameter(self.name, Parameter(weight.detach()))

    def compute_weight(self, module: nn.Module) -> Tensor:
        weight: Parameter = getattr(module, self.name + "_param")
        weight_gain = getattr(module, "_weight_gain")
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
    bias_init: float = 0.0,
    recursive: bool = False,
    reapply: bool = False,
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
                reapply=reapply,
            )
        return module
    if (
        getattr(module, "weighting", None) is not None
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


def _remove_bias_lr_equalizer(
    module: nn.Module,
    recursive: bool = False,
) -> nn.Module:
    if recursive:
        for _module in module.modules():
            _remove_bias_lr_equalizer(_module, recursive=False)
        return module
    if utils.attr_is_value(module, "lr_equalization", True):
        setattr(module, "lr_equalization", False)
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
    if utils.attr_is_value(module, "lr_equalization", True):
        setattr(module, "lr_equalization", False)
    _pop_weight_lr_equalizer(module, raise_exception=False)
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
