from typing import Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _reverse_repeat_tuple

from firewood.common.types import INT
from firewood.functional.resample import nearest_downsample, upsample
from firewood.utils.common import normalize_int_tuple


def firNd(
    input: Tensor, kernel: Tensor, gain: float = 1.0, flip_kernel: bool = False
) -> Tensor:
    rank = input.ndim - 2
    if rank == 1:
        conv = F.conv1d
    elif rank == 2:
        conv = F.conv2d
    elif rank == 3:
        conv = F.conv3d
    else:
        raise ValueError(f"Rank {rank} is not supported.")

    if not flip_kernel:
        kernel = kernel.flip(list(range(kernel.ndim)))

    C = input.size(1)
    if gain != 1.0:
        kernel = kernel * gain ** (kernel.ndim / rank)
    kernel = kernel.view(1, 1, *kernel.shape)
    # kernel = kernel.expand(C, 1, *kernel.shape[2:]) is best implementation.
    # But torch.onnx not support expand before convolution.
    kernel = torch.cat([kernel for _ in range(C)], dim=0)

    if kernel.ndim == input.ndim:
        return conv(input, kernel, groups=C)
    output = input
    for i in range(2, rank + 2):
        output = conv(output, kernel.unsqueeze(i), groups=C)
    return output


def upfirdnNd(
    input: Tensor,
    kernel: Tensor,
    gain: float = 1.0,
    flip_kernel: bool = False,
    up: INT = 1,
    down: INT = 1,
    padding: INT = 0,
    upsample_mode: str = "zeros",
) -> Tensor:
    up = normalize_int_tuple(up, input.ndim - 2)
    down = normalize_int_tuple(down, input.ndim - 2)
    padding = _parse_padding(input.ndim - 2, padding)

    # upsample
    if any(u > 1 for u in up) and all(u >= 1 for u in up):
        input = upsample(input, up, mode=upsample_mode)
    if upsample_mode.startswith("zero"):
        gain = gain * cast(float, np.prod(up))

    # pad
    if any(p != 0 for p in padding):
        input = F.pad(input, padding, mode="constant", value=0)

    # fir
    output = firNd(input, kernel, gain=gain, flip_kernel=flip_kernel)

    # downsample
    if any(d > 1 for d in down) and all(d >= 1 for d in down):
        output = nearest_downsample(output, down)
    return output


def _parse_padding(rank: int, padding: INT) -> Tuple[int, ...]:
    if isinstance(padding, int):
        padding = (padding,) * rank * 2
    elif len(padding) == 1:
        padding = (padding[0],) * rank * 2
    elif len(padding) == rank:
        padding = cast(Tuple[int, ...], _reverse_repeat_tuple(padding, 2))
    elif len(padding) == rank * 2:
        padding = tuple(padding)
    else:
        raise ValueError(
            f"Padding must be either integer or iterable of length {rank} or "
            f"{rank * 2}. Received: {padding}"
        )
    return padding
