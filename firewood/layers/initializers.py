from typing import Callable

import torch.nn.init as init
from torch import Tensor


def get(initializer: str) -> Callable[..., Tensor]:
    initializer = initializer.lower()
    if initializer.startswith("zero"):
        return init.zeros_
    if initializer.startswith("one"):
        return init.ones_
    if initializer == "normal":
        return init.normal_
    if initializer == "uniform":
        return init.uniform_
    if initializer == "constant":
        return init.constant_
    if initializer in {"xavier_uniform", "glorot_uniform"}:
        return init.xavier_uniform_
    if initializer in {"kaiming_uniform", "he_uniform"}:
        return init.kaiming_uniform_
    if initializer in {"xavier", "xavier_normal", "glorot", "glorot_normal"}:
        return init.xavier_normal_
    if initializer in {"kaiming", "kaiming_normal", "he", "he_normal"}:
        return init.kaiming_normal_
    if initializer == "orthogonal":
        return init.orthogonal_
    if initializer == "eye":
        return init.eye_
    if initializer == "dirac":
        return init.dirac_
    if not initializer.endswith("_"):
        initializer += "_"
    if hasattr(init, initializer):
        return getattr(init, initializer)
    raise NotImplementedError(
        f"Received initializer is not implemented. Received: {initializer}"
    )
