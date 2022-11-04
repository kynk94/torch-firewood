import torch
import torch.nn.functional as F
from torch import Tensor

from firewood.common.types import INT
from firewood.utils.common import normalize_int_tuple


def zero_insertion_upsample(input: Tensor, factor: INT) -> Tensor:
    """
    Upsampling by inserting 0.

    Args:
        input: Input tensor. Shape (N, C, ...) (any number of dimensions).
        factor: Upsampling factor.
            If factor is a single integer, it is used for all spatial dims.
            If factor is a tuple, it is used for input spatial dim's order.
            e.g. (2, 3) for upsampling by 2 in height and 3 in width.
    """
    factor = normalize_int_tuple(factor, input.ndim - 2)
    if set(factor) == {1}:
        return input
    if any(f < 1 for f in factor):
        raise ValueError("factor must be integer >= 1")
    prepare_shape = list(input.shape[:2])
    return_shape = list(input.shape[:2])
    pad = []
    for i, f in zip(input.shape[2:], factor):
        prepare_shape.extend([i, 1])
        return_shape.append(i * f)
        pad.extend([0, 0, f - 1, 0])
    reversed_pad = tuple(reversed(pad))

    output = input.view(prepare_shape)
    output = torch._C._VariableFunctions.constant_pad_nd(
        input=output, pad=reversed_pad, value=0.0
    )
    return output.view(return_shape)


def nearest_upsample(input: Tensor, factor: INT) -> Tensor:
    """
    Upsampling by nearest neighbor.
    This implementation is slower than `F.interpolate`.
    """
    factor = normalize_int_tuple(factor, input.ndim - 2)
    if set(factor) == {1}:
        return input
    if any(f < 1 for f in factor):
        raise ValueError("factor must be integer >= 1")
    prepare_shape = list(input.shape[:2])
    expand_shape = list(input.shape[:2])
    return_shape = list(input.shape[:2])
    for i, f in zip(input.shape[2:], factor):
        prepare_shape.extend([i, 1])
        expand_shape.extend([i, f])
        return_shape.append(i * f)
    output = input.view(prepare_shape).expand(expand_shape).contiguous()
    return output.view(return_shape)


def upsample(input: Tensor, factor: INT, mode: str = "zeros") -> Tensor:
    """
    Upsampling by nearest neighbor or inserting 0.

    Args:
        input: Input tensor. Shape (N, C, ...) (any number of dimensions).
        factor: Upsampling factor.
        mode: upsampling mode, "zeros" or "nearest".
    """
    lower_mode = mode.lower()
    if lower_mode.startswith("near"):
        return F.interpolate(input, scale_factor=factor, mode="nearest")
    if lower_mode.startswith("zero"):
        return zero_insertion_upsample(input, factor)
    raise ValueError(f"Unknown upsampling mode: {mode}")


def nearest_downsample(input: Tensor, factor: INT) -> Tensor:
    rank = input.ndim - 2
    factor = normalize_int_tuple(factor, rank)
    if set(factor) == {1}:
        return input
    if any(f < 1 for f in factor):
        raise ValueError("factor must be integer >= 1")

    if rank == 1:
        return input[..., :: factor[0]]
    if rank == 2:
        return input[..., :: factor[0], :: factor[1]]
    if rank == 3:
        return input[..., :: factor[0], :: factor[1], :: factor[2]]
    raise ValueError("Only 1D, 2D, and 3D downsampling is supported.")
