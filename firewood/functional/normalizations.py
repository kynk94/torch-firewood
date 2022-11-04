import math
from typing import Optional, cast

import numpy as np
import torch.linalg as LA
from torch import Tensor

from firewood.common.types import INT


def moment_normalization(
    input: Tensor,
    ord: int = 2,
    dim: Optional[INT] = None,
    use_scaling: bool = True,
    eps: float = 1e-9,
) -> Tensor:
    """
    Return:
        input / (mean(input ** ord, dim) ** (1 / ord) + eps)
    """
    norm = LA.vector_norm(input, ord=ord, dim=dim, keepdim=True)
    if not use_scaling:
        return input / (norm + eps)

    if dim is None:
        dim = range(input.ndim)
    if isinstance(dim, int):
        numel = input.size(dim)
    else:
        numel = cast(int, np.prod(tuple(input.size(d) for d in dim)))
    return input / (norm / math.sqrt(numel) + eps)


def maximum_normalization(
    input: Tensor,
    dim: Optional[INT] = None,
    use_scaling: bool = True,
    eps: float = 1e-9,
) -> Tensor:
    """
    Return:
        input / (max(abs(input), except dim) + eps)
    """
    return moment_normalization(
        input, ord=float("inf"), dim=dim, use_scaling=use_scaling, eps=eps
    )
