import os
import random
from typing import Any, Optional

import numpy as np
import torch
from torch.autograd.grad_mode import _DecoratorContextManager

torch.backends.cudnn.benchmark = True
_lr_equalization = False
_weight_grad_disabled = False
_runtime_build = False
_seed: Optional[int] = None


def lr_equalization() -> bool:
    return _lr_equalization


def set_lr_equalization(value: bool = False) -> bool:
    if not isinstance(value, bool):
        raise TypeError("lr_equalization must be bool")
    global _lr_equalization
    _lr_equalization = value
    return _lr_equalization


def weight_grad_disabled() -> bool:
    return _weight_grad_disabled


def set_weight_grad_disabled(value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError("weight_grad_disabled must be bool")
    global _weight_grad_disabled
    _weight_grad_disabled = value
    return _weight_grad_disabled


class no_weight_grad_in_gfix_conv(_DecoratorContextManager):
    """
    Forcefully disable computation of gradients with respect to the weights of
    GFixConvNd layers.

    If `backend.set_weight_grad_disabled(True)`, the gradients of weight will be
    `None`, even outside the `no_weight_grad_in_gfix_conv` context. Otherwise,
    the gradients of weight will be `None` only entering the context.

    The context is used by some regularization algorithms like path-length reg.
    """

    def __init__(self) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self.prev = False

    def __enter__(self) -> None:
        self.prev = weight_grad_disabled()
        set_weight_grad_disabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        set_weight_grad_disabled(self.prev)


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
    if _runtime_build:
        # prevent circular import
        from firewood.utils.extensions import CUDAExtension

        CUDAExtension.import_C()
    return _runtime_build


def seed() -> Optional[int]:
    return _seed


def set_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    elif not isinstance(seed, int):
        raise TypeError("seed must be int")
    global _seed
    _seed = seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import pytorch_lightning as pl

        pl.seed_everything(seed)
    except ImportError:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return _seed
