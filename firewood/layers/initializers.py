import functools
import warnings
from typing import Any, Callable, Union

import torch.nn.init as init
from torch import Tensor


def get(
    initializer: Union[str, int, float], **kwargs: Any
) -> Callable[..., Tensor]:
    if isinstance(initializer, (int, float)):
        return functools.partial(init.constant_, val=initializer)
    initializer = initializer.lower()
    if initializer.startswith("zero"):
        return init.zeros_
    if initializer.startswith("one"):
        return init.ones_
    if initializer == "eye":
        return init.eye_
    if initializer == "normal":
        return functools.partial(init.normal_, **kwargs)
    if initializer == "uniform":
        return functools.partial(init.uniform_, **kwargs)
    if initializer == "constant":
        return functools.partial(init.constant_, **kwargs)
    if initializer in {"xavier_uniform", "glorot_uniform"}:
        return functools.partial(init.xavier_uniform_, **kwargs)
    if initializer in {"kaiming_uniform", "he_uniform"}:
        return functools.partial(init.kaiming_uniform_, **kwargs)
    if initializer in {"xavier", "xavier_normal", "glorot", "glorot_normal"}:
        return functools.partial(init.xavier_normal_, **kwargs)
    if initializer in {"kaiming", "kaiming_normal", "he", "he_normal"}:
        return functools.partial(init.kaiming_normal_, **kwargs)
    if initializer == "orthogonal":
        return functools.partial(init.orthogonal_, **kwargs)
    if initializer == "dirac":
        return functools.partial(init.dirac_, **kwargs)

    if not initializer.endswith("_"):
        initializer += "_"
    if hasattr(init, initializer):
        return functools.partial(getattr(init, initializer), **kwargs)
    raise NotImplementedError(
        f"Received initializer is not implemented. Received: {initializer}"
    )
