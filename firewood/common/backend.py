import os
import random
from typing import Optional

import numpy as np
import torch

_lr_equalization = False
_weight_gradients_disabled = False
_runtime_build = False
_seed: Optional[int] = None


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
