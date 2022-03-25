_lr_equalization = False
_weight_gradients_disabled = False
_runtime_build = False


def lr_equalization() -> bool:
    return _lr_equalization


def set_lr_equalization(lr_equalization: bool = False) -> bool:
    if not isinstance(lr_equalization, bool):
        raise TypeError("lr_equalization must be bool")
    global _lr_equalization
    _lr_equalization = lr_equalization
    return _lr_equalization


def weight_gradients_disabled() -> bool:
    return _weight_gradients_disabled


def set_conv_weight_gradients_disabled(disable_weight_gradients: bool) -> bool:
    if not isinstance(disable_weight_gradients, bool):
        raise TypeError("disable_weight_gradients must be bool")
    global _weight_gradients_disabled
    _weight_gradients_disabled = disable_weight_gradients
    return _weight_gradients_disabled


def runtime_build() -> bool:
    return _runtime_build


def set_runtime_build(runtime_build: bool) -> bool:
    """
    When runtime_build is True, csrc will be built at runtime.
    Default is False.
    """
    if not isinstance(runtime_build, bool):
        raise TypeError("runtime_build must be bool")
    global _runtime_build
    _runtime_build = runtime_build
    return _runtime_build
