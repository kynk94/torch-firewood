import functools
from typing import Any, Callable, Optional

import torch.nn as nn
from torch.nn.utils import spectral_norm, weight_norm

from firewood import utils
from firewood.hooks.weight_denormalizations import weight_denorm


def get(
    normalization: str,
    n_power_iterations: int = 1,
    demodulate: bool = True,
    eps: float = 1e-8,
    **kwargs: Any,
) -> Optional[Callable[..., nn.Module]]:
    if normalization is None:
        return None
    normalization = normalization.lower()
    if normalization.startswith("spectral"):
        return functools.partial(
            spectral_norm,
            n_power_iterations=n_power_iterations,
            eps=eps,
            **kwargs,
        )
    if normalization in {"weight", "weight_norm", "weight_normalization"}:
        return functools.partial(weight_norm, **kwargs)
    if normalization in {
        "demodulation",
        "weight_demodulation",
        "dn",
        "denorm",
        "weight_denorm",
        "weight_denormalization",
    }:
        modulation_features = utils.search_kwargs(
            kwargs, ("modulation_features", "mf"), None, pop=True
        )
        if modulation_features is None:
            raise ValueError("modulation_features must be specified.")
        pre_normalize = utils.search_kwargs(
            kwargs, ("pre_normalize", "pn"), "maximum", pop=True
        )
        return functools.partial(
            weight_denorm,
            modulation_features=modulation_features,
            demodulate=demodulate,
            pre_normalize=pre_normalize,
            eps=eps,
            **kwargs,
        )
    raise ValueError(f"Unknown weight normalization: {normalization}")
