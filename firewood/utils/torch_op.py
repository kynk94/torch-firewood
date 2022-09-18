from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair, _single, _triple

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
        return args_to(
            *arg, dtype=dtype, device=device, non_blocking=non_blocking
        )
    return arg


def args_to(
    *args: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DEVICE] = None,
    non_blocking: bool = True,
) -> Any:
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
        module, ["in_features", "in_channels"]
    )
    out_features: Optional[int] = search_attr(
        module, ["out_features", "out_channels"]
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


def unsqueeze_view(tensor: Tensor, dim: int, times: int) -> Tensor:
    if dim < 0:
        dim += tensor.ndim + 1
    return tensor.view(tensor.shape[:dim] + (1,) * times + tensor.shape[dim:])


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
