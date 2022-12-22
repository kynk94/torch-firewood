from typing import Any, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import (
    _pair,
    _reverse_repeat_tuple,
    _single,
    _triple,
)

from firewood.common.types import DEVICE, INT, SAME_PADDING
from firewood.utils.common import search_attr


def _arg_to(
    arg: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DEVICE] = None,
    non_blocking: bool = True,
) -> Any:
    if isinstance(arg, Tensor):
        if device is not None and not is_cuda(device):
            arg = arg.detach()
        return arg.to(dtype=dtype, device=device, non_blocking=non_blocking)
    if isinstance(arg, dict):
        return kwargs_to(
            **arg, dtype=dtype, device=device, non_blocking=non_blocking
        )
    if isinstance(arg, Sequence):
        return tuple(
            _arg_to(_arg, dtype=dtype, device=device, non_blocking=non_blocking)
            for _arg in arg
        )
    return arg


def args_to(
    *args: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DEVICE] = None,
    non_blocking: bool = True,
) -> Any:
    if isinstance(args, tuple) and len(args) == 1:
        return _arg_to(args[0], dtype, device, non_blocking)
    outputs = []
    for arg in args:
        outputs.append(_arg_to(arg, dtype, device, non_blocking))
    return tuple(outputs)


def kwargs_to(
    _kwargs: Optional[dict] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DEVICE] = None,
    non_blocking: bool = True,
    **kwargs: Any,
) -> Any:
    if _kwargs:
        kwargs.update(_kwargs)
        return kwargs_to(
            **kwargs, dtype=dtype, device=device, non_blocking=non_blocking
        )
    outputs = dict()
    for key, arg in kwargs.items():
        outputs[key] = _arg_to(arg, dtype, device, non_blocking)
    return outputs


def is_cuda(device: Union[torch.device, int, str, None]) -> bool:
    if device is None:
        return False
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    return device.type == "cuda"


def get_rank(module: nn.Module) -> int:
    rank: Optional[int] = getattr(module, "rank", None)
    if rank is not None:
        return rank
    weight: Optional[Tensor] = getattr(module, "weight", None)
    if weight is not None:
        return weight.ndim - 2
    raise ValueError(f"Could not find `rank` in {module}.")


def get_in_out_features(module: nn.Module) -> Tuple[int, int]:
    in_features: Optional[int] = search_attr(
        module, ("in_features", "in_channels"), None
    )
    out_features: Optional[int] = search_attr(
        module, ("out_features", "out_channels"), None
    )
    if in_features is None or out_features is None:
        weight: Optional[Tensor] = getattr(module, "weight", None)
        if weight is None:
            raise ValueError(
                f"Could not find `in_features` or `out_features` in {module}."
            )
        out_features, in_features = weight.shape[:2]
    return in_features, out_features


def unsqueeze_repeatedly(
    tensor: Tensor, dim: int, times: int, inplace: bool = False
) -> Tensor:
    if inplace:
        for _ in range(times):
            tensor.unsqueeze_(dim)
        return tensor
    for _ in range(times):
        tensor = tensor.unsqueeze(dim)
    return tensor


def unsqueeze_view(tensor: Tensor, dim: int, times: int = 1) -> Tensor:
    if dim < 0:
        dim += tensor.ndim + 1
    return tensor.view(tensor.shape[:dim] + (1,) * times + tensor.shape[dim:])


def clone_to_cpu_tensor(tensor: Tensor) -> Tensor:
    if tensor.device.type == "cpu":
        return tensor.clone()
    return tensor.detach().cpu()


def conv_same_padding_for_functional_pad(
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    dilation: Tuple[int, ...],
    spatial_size: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, ...]:
    """
    Calculate padding size for same padding in functional.pad.

    The padding size is dependent on the input `spatial_size`, so it is not
    possible to calculate the padding size in module's init method.
    """
    if spatial_size is None:
        if set(stride) != {1}:
            raise ValueError(
                "Cannot calculate padding size for stride != 1 "
                f"without `spatial_size`. Got {stride}."
            )
        spatial_size = (1,) * len(kernel_size)
    pad = []
    for k, d, s, i in zip(kernel_size, dilation, stride, spatial_size):
        div, mod = divmod((k - 1) * d + 1 - s + i % s, 2)
        pad.extend([div + mod, div])
    return tuple(reversed(pad))


def conv_transpose_same_padding_for_functional_pad(
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    PyTorch's output_padding cannot be None, so ignore it for same padding.
    Output size with same padding:
    >>> O = stride * input_size + output_padding

    In Tensorflow, output size with same padding is following
    >>> if output_padding is None:
            O = stride * input_size
        else:
            F = (kernel_size - 1) * dilation + 1
            O = stride * (input_size - 1) + F % 2 + output_padding

    If Tensorflow's output_padding is not None, the padding is following
    >>> pad = []
        for k, d in zip(kernel_size, dilation):
            div, mod = divmod((k - 1) * d, 2)
            pad.extend([div + mod, div])
        pad = tuple(reversed(pad))
    """
    pad = []
    for k, d, s in zip(kernel_size, dilation, stride):
        div, mod = divmod((k - 1) * d + 1 - s, 2)
        pad.extend([div + mod, div])
    return tuple(reversed(pad))


def padding_for_functional_pad(rank: int, padding: INT) -> Tuple[int, ...]:
    """
    Convert padding to a tuple of length 2 * rank for `torch.nn.functional.pad`.

    Examples:
        >>> padding_for_functional_pad(2, 1)
        (1, 1, 1, 1)
        >>> padding_for_functional_pad(2, (1, 2))
        (2, 2, 1, 1)
        >>> padding_for_functional_pad(2, (1, 2, 3, 4))
        (3, 4, 1, 2)
    """
    if isinstance(padding, int):
        return (padding,) * rank * 2
    if len(padding) == rank:
        return cast(Tuple[int, ...], _reverse_repeat_tuple(padding, 2))
    if len(padding) == rank * 2:
        return tuple(
            p
            for i in range(rank - 1, -1, -1)
            for p in padding[i * 2 : (i + 1) * 2]
        )
    raise ValueError("Invalid padding: {}".format(padding))


def _single_padding(
    obj: Union[SAME_PADDING, INT]
) -> Union[SAME_PADDING, Tuple[int]]:
    if isinstance(obj, str):
        if obj.lower() != "same":
            raise ValueError(f"Unknown padding: {obj}")
        return obj
    return _single(obj)


def _pair_padding(
    obj: Union[SAME_PADDING, INT]
) -> Union[SAME_PADDING, Tuple[int, int]]:
    if isinstance(obj, str):
        if obj.lower() != "same":
            raise ValueError(f"Unknown padding: {obj}")
        return obj
    return _pair(obj)


def _triple_padding(
    obj: Union[SAME_PADDING, INT]
) -> Union[SAME_PADDING, Tuple[int, int, int]]:
    if isinstance(obj, str):
        if obj.lower() != "same":
            raise ValueError(f"Unknown padding: {obj}")
        return obj
    return _triple(obj)
