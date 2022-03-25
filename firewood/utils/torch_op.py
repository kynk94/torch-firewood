import re
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair, _single, _triple

from firewood.common.types import DEVICE, INT, SAME_PADDING
from firewood.utils.common import search_attr


def args_to(
    *args: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DEVICE] = None,
    non_blocking: bool = True,
) -> Any:
    outputs = []
    for arg in args:
        if isinstance(arg, Sequence):
            arg = args_to(
                *arg, dtype=dtype, device=device, non_blocking=non_blocking
            )
        elif isinstance(arg, dict):
            arg = kwargs_to(
                **arg, dtype=dtype, device=device, non_blocking=non_blocking
            )
        elif isinstance(arg, Tensor):
            if device is not None and not is_cuda(device):
                arg = arg.detach()
            arg = arg.to(dtype=dtype, device=device, non_blocking=non_blocking)
        outputs.append(arg)
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
        if isinstance(arg, Sequence):
            arg = args_to(
                *arg, dtype=dtype, device=device, non_blocking=non_blocking
            )
        elif isinstance(arg, dict):
            arg = kwargs_to(
                **arg, dtype=dtype, device=device, non_blocking=non_blocking
            )
        elif isinstance(arg, Tensor):
            if device is not None and not is_cuda(device):
                arg = arg.detach()
            arg = arg.to(dtype=dtype, device=device, non_blocking=non_blocking)
        outputs[key] = arg
    return outputs


def is_cuda(device: Union[torch.device, int, str, None]) -> bool:
    if device is None:
        return False
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    return device.type == "cuda"


def normalize_activation_name(activation: Optional[str]) -> str:
    if activation is None or len(activation) == 0:
        return "linear"
    activation = activation.lower()
    if activation == "lrelu":
        return "leaky_relu"
    if activation == "swish":
        return "silu"
    return activation


def normalize_op_order(op_order: str) -> str:
    op_order = op_order.upper()
    if len(op_order) != 3:
        raise ValueError("Op order must be 3 characters long.")
    op_order = re.sub("[^ANW]", "W", op_order)
    for char in "ANW":
        if op_order.count(char) != 1:
            raise ValueError(
                f"Op order must contain exactly one {char} character."
            )
    return op_order


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
                f"Could not find in_features or out_features in {module}."
            )
        out_features, in_features = weight.shape[:2]
    return in_features, out_features


def _single_padding(
    obj: Union[SAME_PADDING, INT]
) -> Union[SAME_PADDING, Tuple[int]]:
    if isinstance(obj, str):
        return obj
    return _single(obj)


def _pair_padding(
    obj: Union[SAME_PADDING, INT]
) -> Union[SAME_PADDING, Tuple[int, int]]:
    if isinstance(obj, str):
        return obj
    return _pair(obj)


def _triple_padding(
    obj: Union[SAME_PADDING, INT]
) -> Union[SAME_PADDING, Tuple[int, int, int]]:
    if isinstance(obj, str):
        return obj
    return _triple(obj)
