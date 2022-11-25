from typing import Tuple, Union

import torch
from torch import Size, Tensor
from torch.nn.grad import conv1d_weight, conv2d_weight, conv3d_weight

from firewood.common.types import INT

_CUDNN_FLAGS = [
    torch.backends.cudnn.benchmark,
    torch.backends.cudnn.deterministic,
    torch.backends.cudnn.allow_tf32,
]


def conv_weight_cudnn(
    input: Tensor,
    weight_size: Union[Size, Tuple[int, ...]],
    grad_output: Tensor,
    stride: INT = 1,
    padding: INT = 0,
    dilation: INT = 1,
    groups: int = 1,
) -> Tensor:
    """
    support conv2d, conv3d

    deprecated in torch 1.11
    """
    operation_name = "aten::cudnn_convolution_backward_weight"
    return torch._C._jit_get_operation(operation_name)(  # type: ignore
        weight_size,
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        *_CUDNN_FLAGS,
    )


def conv_transpose_weight_cudnn(
    input: Tensor,
    weight_size: Union[Size, Tuple[int, ...]],
    grad_output: Tensor,
    stride: INT = 1,
    padding: INT = 0,
    dilation: INT = 1,
    groups: int = 1,
) -> Tensor:
    """
    support conv_transpose2d, conv_transpose3d

    deprecated in torch 1.11
    """
    operation_name = "aten::cudnn_convolution_transpose_backward_weight"
    return torch._C._jit_get_operation(operation_name)(  # type: ignore
        weight_size,
        grad_output,
        input,
        padding,
        stride,
        dilation,
        groups,
        *_CUDNN_FLAGS,
    )


def conv_transpose1d_weight(
    input: Tensor,
    weight_size: Union[Size, Tuple[int, ...]],
    grad_output: Tensor,
    stride: INT = 1,
    padding: INT = 0,
    dilation: INT = 1,
    groups: int = 1,
) -> Tensor:
    return conv1d_weight(
        grad_output, weight_size, input, stride, padding, dilation, groups
    )


def conv_transpose2d_weight(
    input: Tensor,
    weight_size: Union[Size, Tuple[int, ...]],
    grad_output: Tensor,
    stride: INT = 1,
    padding: INT = 0,
    dilation: INT = 1,
    groups: int = 1,
) -> Tensor:
    return conv2d_weight(
        grad_output, weight_size, input, stride, padding, dilation, groups
    )


def conv_transpose3d_weight(
    input: Tensor,
    weight_size: Union[Size, Tuple[int, ...]],
    grad_output: Tensor,
    stride: INT = 1,
    padding: INT = 0,
    dilation: INT = 1,
    groups: int = 1,
) -> Tensor:
    return conv3d_weight(
        grad_output, weight_size, input, stride, padding, dilation, groups
    )
