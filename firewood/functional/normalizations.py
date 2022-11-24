from typing import Optional, Union

import torch.linalg as LA
from torch import Tensor

from firewood.common.types import INT


def moment_normalization(
    input: Tensor,
    ord: Union[float, int] = 2,
    dim: Optional[INT] = None,
    mean: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """
    Return:
        if mean is True (default):
            input / (mean(abs(input) ** ord, dim) ** (1 / ord) + eps)
        if mean is False:
            input / (sum(abs(input) ** ord, dim) ** (1 / ord) + eps)
    """
    if isinstance(dim, range):
        dim = tuple(dim)

    norm = LA.vector_norm(input, ord=ord, dim=dim, keepdim=True) + eps
    output = input / norm
    if not mean or ord == float("inf"):
        return output

    if dim is None:
        dim = range(input.ndim)
    if isinstance(dim, int):
        numel = input.size(dim)
    else:
        numel = 1
        for d in dim:
            numel *= input.size(d)
    return output / numel ** (1 / ord)


def maximum_normalization(
    input: Tensor,
    dim: Optional[INT] = None,
    mean: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """
    Return:
        input / (max(abs(input), except dim) + eps)
    """
    return moment_normalization(
        input, ord=float("inf"), dim=dim, mean=mean, eps=eps
    )
