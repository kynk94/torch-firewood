import contextlib
import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as grad
import torch.nn.init as init
from torch import Size, Tensor
from torch.nn.modules.utils import (
    _pair,
    _reverse_repeat_tuple,
    _single,
    _triple,
)
from torch.nn.parameter import Parameter

from firewood import utils
from firewood.common.backend import (
    set_conv_weight_gradients_disabled,
    weight_gradients_disabled,
)
from firewood.common.constant import NULL_TENSOR
from firewood.common.types import DEVICE, INT, SAME_PADDING
from firewood.layers import initializers
from firewood.utils import _pair_padding, _single_padding, _triple_padding

# Forcefully disable computation of gradients with respect to the weights.
#
# If True, the gradients of weight will be None, even outside the
# no_weight_gradients_in_gfix_conv context. Otherwise, the gradients of weight
# will be None only in the no_weight_gradients_in_gfix_conv context.
#
# The context is used by some regularization algorithms like path-length reg.


@contextlib.contextmanager
def no_weight_gradients_in_gfix_conv() -> Any:
    old = weight_gradients_disabled()
    set_conv_weight_gradients_disabled(True)
    yield
    set_conv_weight_gradients_disabled(old)


_CONV_CUSTOM_GRAD_CACHE: Dict[
    Tuple[
        bool,  # transposed
        Tuple[int, ...],  # weight_shape
        Tuple[int, ...],  # stride
        Tuple[int, ...],  # padding
        Tuple[int, ...],  # dilation
        Tuple[int, ...],  # output_padding
        int,  # groups
        str,  # device type
    ],
    Type[torch.autograd.Function],
] = dict()

_CUDNN_FLAGS = [
    torch.backends.cudnn.benchmark,
    torch.backends.cudnn.deterministic,
    torch.backends.cudnn.allow_tf32,  # type: ignore
]


class _GFixConvNd(nn.Module):
    """
    Support gradient fix function for convolutional layers.
    padding:
        same: which make output has the same size as the input.
        int: up & boddom & left & right
        (int, int): up & bottom, left & right
        (int, int, int, int): up, bottom, left, right
    """

    _groups: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Union[SAME_PADDING, INT],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: Union[str, int, float],
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
        device: Optional[DEVICE] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.device = torch.device(device or "cpu")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.rank = len(self.kernel_size)

        if self.transposed:
            weight_shape = (
                in_channels,
                out_channels // groups,
                *kernel_size,
            )
        else:
            weight_shape = (
                out_channels,
                in_channels // groups,
                *kernel_size,
            )
        self.weight_shape = weight_shape
        self.weight = Parameter(
            torch.empty(self.weight_shape, dtype=dtype, device=self.device)
        )

        if bias:
            self.bias = Parameter(
                torch.empty(out_channels, dtype=dtype, device=self.device)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # set padding
        self.conv_pad = False
        self.conv_transpose_pad = False
        self.padding_value = 0.0
        # check padding_mode
        if isinstance(padding_mode, (int, float)):
            self.conv_pad = padding_mode != 0
            self.padding_value, self.padding_mode = padding_mode, "constant"
        elif padding_mode == "zeros":
            self.padding_mode = "constant"
        else:
            self.conv_pad = not self.transposed
            self.padding_mode = padding_mode

        if isinstance(padding, str):
            padding = padding.lower()  # type: ignore
            if padding != "same":
                raise ValueError("Only 'same' padding is supported")
            padding = _calc_same_padding(
                transposed=self.transposed,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
            )
        else:
            padding = _calc_padding(rank=self.rank, padding=padding)
        self.padding = padding

        if len(self.padding) == self.rank:
            if self.padding_mode == "constant" and self.padding_value == 0.0:
                self.conv_pad = False
            else:
                self.padding = _reverse_repeat_tuple(self.padding, self.rank)
            self.conv_transpose_pad = False

        if len(self.padding) == self.rank * 2:
            self.conv_pad = not self.transposed
            self.conv_transpose_pad = self.transposed

        self.to(device=self.device)

    def reset_parameters(self) -> None:
        # weight initialization
        # add other init methods here if needed
        if self.weight_initializer in {
            "kaiming_uniform",
            "he_uniform",
        }:  # default torch init
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            initializers.get(self.weight_initializer)(self.weight)

        if self.bias is None:
            return

        # bias initialization
        # add other init methods here if needed
        if self.bias_initializer == "uniform":  # default torch init
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            initializers.get(self.bias_initializer)(self.bias)

    def _apply(self, fn: Callable[..., Any]) -> "_GFixConvNd":
        if "t" in fn.__code__.co_varnames:
            with torch.no_grad():
                device = getattr(fn(NULL_TENSOR), "device", "cpu")
            self.device = torch.device(device)
        return super()._apply(fn)

    @property
    def groups(self) -> int:
        """
        `groups` is modified during weight modulation (weight denorm) operation,
        so need to manage as a property.
        See `firewood.layers.weight_normalizations.py` for more details.
        """
        return self._groups

    @groups.setter
    def groups(self, value: int) -> None:
        if self.groups == value:
            return
        self._groups = value
        if getattr(self, "weight_shape", None) is None:
            return
        self.weight_shape = (
            self.out_channels * self.groups,
            *self.weight_shape[1:],
        )

    @property
    def operation(self) -> Callable[..., Tensor]:
        if self.conv_pad or self.conv_transpose_pad:
            padding = (0,) * self.rank
        else:
            padding = self.padding
        return _load_operation(
            transposed=self.transposed,
            weight_shape=self.weight_shape,
            stride=self.stride,
            padding=padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            device=self.device,
        )

    def forward(self, input: Tensor) -> Tensor:
        if self.conv_pad:
            # padding
            input = F.pad(
                input=input,
                pad=self.padding,
                mode=self.padding_mode,
                value=self.padding_value,
            )
        output = self.operation(input, self.weight, self.bias)
        if self.conv_transpose_pad:
            # cropping
            output = F.pad(
                input=output,
                pad=tuple(-p for p in self.padding),
                mode="constant",
            )
        return output

    def extra_repr(self) -> str:
        s = []
        s.extend(
            [
                f"{self.in_channels}",
                f"{self.out_channels}",
                f"kernel_size={self.kernel_size}",
                f"stride={self.stride}",
            ]
        )
        if self.padding != (0,) * len(self.padding):
            s.append(f"padding={self.padding}")
        if self.dilation != (1,) * len(self.dilation):
            s.append(f"dilation={self.dilation}")
        if self.output_padding != (0,) * len(self.output_padding):
            s.append(f"output_padding={self.output_padding}")
        if self.groups != 1:
            s.append(f"groups={self.groups}")
        s.append(f"bias={self.bias is not None}")
        if self.padding_mode == "constant":
            s.append(f"padding_value={self.padding_value}")
        else:
            s.append(f"padding_mode={self.padding_mode}")
        s.extend(
            [
                f"weight_initializer={self.weight_initializer}",
                f"bias_initializer={self.bias_initializer}",
            ]
        )
        return ", ".join(s)


class GFixConv1d(_GFixConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = _single(1),
        padding: Union[SAME_PADDING, INT] = _single(0),
        dilation: INT = _single(1),
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single_padding(padding),
            dilation=_single(dilation),
            transposed=False,
            output_padding=_single(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class GFixConv2d(_GFixConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = _pair(1),
        padding: Union[SAME_PADDING, INT] = _pair(0),
        dilation: INT = _pair(1),
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair_padding(padding),
            dilation=_pair(dilation),
            transposed=False,
            output_padding=_pair(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class GFixConv3d(_GFixConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = _triple(1),
        padding: Union[SAME_PADDING, INT] = _triple(0),
        dilation: INT = _triple(1),
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple_padding(padding),
            dilation=_triple(dilation),
            transposed=False,
            output_padding=_triple(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class GFixConvTranspose1d(_GFixConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = _single(1),
        padding: Union[SAME_PADDING, INT] = _single(0),
        dilation: INT = _single(1),
        output_padding: INT = _single(0),
        groups: int = 1,
        bias: bool = False,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_single(kernel_size),
            stride=_single(stride),
            padding=_single_padding(padding),
            dilation=_single(dilation),
            transposed=True,
            output_padding=_single(output_padding),
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class GFixConvTranspose2d(_GFixConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = _pair(1),
        padding: Union[SAME_PADDING, INT] = _pair(0),
        dilation: INT = _pair(1),
        output_padding: INT = _pair(0),
        groups: int = 1,
        bias: bool = False,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_pair(kernel_size),
            stride=_pair(stride),
            padding=_pair_padding(padding),
            dilation=_pair(dilation),
            transposed=True,
            output_padding=_pair(output_padding),
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class GFixConvTranspose3d(_GFixConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = _triple(1),
        padding: Union[SAME_PADDING, INT] = _triple(0),
        dilation: INT = _triple(1),
        output_padding: INT = _triple(0),
        groups: int = 1,
        bias: bool = False,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=_triple(kernel_size),
            stride=_triple(stride),
            padding=_triple_padding(padding),
            dilation=_triple(dilation),
            transposed=True,
            output_padding=_triple(output_padding),
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


def _calc_same_padding(
    transposed: bool,
    kernel_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> Tuple[int, ...]:
    pad = []
    if transposed:
        for k, d, s in zip(kernel_size, dilation, stride):
            div, mod = divmod((k - 1) * d + 1 - s, 2)
            pad.extend([div + mod, div])
    else:
        for k, d in zip(kernel_size, dilation):
            div, mod = divmod((k - 1) * d, 2)
            pad.extend([div + mod, div])
    padding = tuple(reversed(pad))
    if len(set(padding)) == 1:
        padding = (padding[0],) * len(kernel_size)
    return padding


def _calc_padding(rank: int, padding: INT) -> Tuple[int, ...]:
    if isinstance(padding, int):
        return (padding,) * rank
    if len(padding) == rank:
        padding = cast(Tuple[int, ...], _reverse_repeat_tuple(padding, 2))
    elif len(padding) == rank * 2:
        reversed_pad = [
            padding[i * 2 : (i + 1) * 2] for i in range(rank - 1, -1, -1)
        ]
        flatten_pad = tuple(itertools.chain.from_iterable(reversed_pad))
        padding = flatten_pad
    else:
        raise ValueError("Invalid padding: {}".format(padding))
    if len(set(padding)) == 1:
        padding = (padding[0],) * rank
    return padding


def _calc_output_padding(
    input_shape: Tuple[int, ...],
    output_shape: Optional[Tuple[int, ...]] = None,
    kernel_size: INT = 1,
    stride: INT = 1,
    padding: INT = 0,
    output_padding: INT = 0,
    dilation: INT = 1,
) -> Tuple[int, ...]:
    rank = len(input_shape) - 2
    if output_shape is None:
        return _single(output_padding)
    kernel_size = utils.normalize_int_tuple(kernel_size, rank)
    stride = utils.normalize_int_tuple(stride, rank)
    if isinstance(padding, int):
        padding = utils.normalize_int_tuple(padding, rank)
    else:
        padding = cast(Tuple[int, ...], padding)
    dilation = utils.normalize_int_tuple(dilation, rank)

    k = len(input_shape) - 2
    if len(output_shape) == k + 2:
        output_shape = output_shape[2:]
    if len(output_shape) != k:
        raise ValueError(
            "output_shape must have {} or {} elements (got {})".format(
                k, k + 2, len(output_shape)
            )
        )

    min_sizes = torch.jit.annotate(List[int], [])
    max_sizes = torch.jit.annotate(List[int], [])
    for d in range(k):
        _dilation = 1 if dilation is None else dilation[d]
        dim_size = (
            (input_shape[d + 2] - 1) * stride[d]
            - 2 * padding[d]
            + _dilation * (kernel_size[d] - 1)
            + 1
        )
        min_sizes.append(dim_size)
        max_sizes.append(min_sizes[d] + stride[d] - 1)
    for i in range(len(output_shape)):
        size = output_shape[i]
        min_size = min_sizes[i]
        max_size = max_sizes[i]
        if size < min_size or size > max_size:
            raise ValueError(
                (
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})"
                ).format(output_shape, min_sizes, max_sizes, input_shape[2:])
            )

    output_padding = []
    for d in range(k):
        output_padding.append(output_shape[d] - min_sizes[d])
    return tuple(output_padding)


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
    """
    operation = torch._C._jit_get_operation(
        "aten::cudnn_convolution_backward_weight"
    )
    return operation(
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
    """
    operation = torch._C._jit_get_operation(
        "aten::cudnn_convolution_transpose_backward_weight"
    )
    return operation(
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
    return grad.conv1d_weight(
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
    return grad.conv2d_weight(
        grad_output, weight_size, input, stride, padding, dilation, groups
    )


def conv_transpose3d_wiehgt(
    input: Tensor,
    weight_size: Union[Size, Tuple[int, ...]],
    grad_output: Tensor,
    stride: INT = 1,
    padding: INT = 0,
    dilation: INT = 1,
    groups: int = 1,
) -> Tensor:
    return grad.conv3d_weight(
        grad_output, weight_size, input, stride, padding, dilation, groups
    )


def _load_operation(
    transposed: bool,
    weight_shape: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    output_padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int,
    device: torch.device,
) -> Callable[..., Tensor]:
    cache_key = (
        transposed,
        weight_shape,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
        device.type,
    )
    if cache_key in _CONV_CUSTOM_GRAD_CACHE:
        return _CONV_CUSTOM_GRAD_CACHE[cache_key].apply

    rank = len(weight_shape) - 2
    kernel_size = weight_shape[2:]
    sum_dim = [0] + list(range(2, rank + 2))

    conv_kwargs = {
        "stride": stride,
        "dilation": dilation,
        "groups": groups,
        "padding": padding,
    }

    conv_operation: Callable[..., Tensor]
    if transposed:
        conv_kwargs.update(output_padding=output_padding)
        if rank == 1:
            conv_operation = F.conv_transpose1d
            weight_operation = conv_transpose1d_weight
        elif rank == 2:
            conv_operation = F.conv_transpose2d
            if utils.is_cuda(device):
                weight_operation = conv_transpose_weight_cudnn
            else:
                weight_operation = conv_transpose2d_weight
        elif rank == 3:
            conv_operation = F.conv_transpose3d
            if utils.is_cuda(device):
                weight_operation = conv_transpose_weight_cudnn
            else:
                weight_operation = conv_transpose3d_wiehgt
    else:
        if rank == 1:
            conv_operation = F.conv1d
            weight_operation = grad.conv1d_weight
        elif rank == 2:
            conv_operation = F.conv2d
            if utils.is_cuda(device):
                weight_operation = conv_weight_cudnn
            else:
                weight_operation = grad.conv2d_weight
        elif rank == 3:
            conv_operation = F.conv3d
            if utils.is_cuda(device):
                weight_operation = conv_weight_cudnn
            else:
                weight_operation = grad.conv3d_weight

    class GFixConvNd(torch.autograd.Function):
        @staticmethod
        # type: ignore[override]
        def forward(
            ctx: Any,
            input: Tensor,
            weight: Tensor,
            bias: Optional[Tensor] = None,
        ) -> Tensor:
            ctx.save_for_backward(
                input if weight.requires_grad else NULL_TENSOR,
                weight if input.requires_grad else NULL_TENSOR,
            )
            ctx.input_shape = input.shape
            output = conv_operation(input, weight, bias, **conv_kwargs)
            return output

        @staticmethod
        # type: ignore[override]
        def backward(
            ctx: Any, grad_output: Tensor
        ) -> Tuple[Optional[Tensor], ...]:
            input, weight = ctx.saved_tensors
            input_shape = ctx.input_shape
            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:
                # If current state is 'not transposed' then the backprop is
                # 'transposed', so output padding is required in 'not
                # transposed' state.
                if transposed:
                    _output_padding = (0,) * rank
                else:
                    _output_padding = _calc_output_padding(
                        input_shape=grad_output.shape,
                        output_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                    )
                # Explicit cast because autocast doesn't work properly.
                grad_input = _load_operation(
                    transposed=not transposed,
                    output_padding=_output_padding,
                    weight_shape=weight_shape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    device=device,
                )(grad_output, weight.to(grad_output.dtype), None)
                if grad_input.shape != input_shape:
                    raise ValueError(
                        "grad_input shape mismatch in backward of GFixConvNd "
                        f"(input: {input_shape}, grad_input: {grad_input.shape})"
                    )

            if ctx.needs_input_grad[1] and not weight_gradients_disabled():
                # Explicit cast because autocast doesn't work properly.
                grad_weight = GFixConvNdGradWeight.apply(
                    grad_output, input.to(grad_output.dtype)
                )
                if grad_weight.shape != weight_shape:
                    raise ValueError(
                        "grad_weight shape mismatch in backward of GFixConvNd "
                        f"(weight: {weight_shape}, grad_weight: {grad_weight.shape})"
                    )

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(sum_dim)

            return grad_input, grad_weight, grad_bias

    class GFixConvNdGradWeight(torch.autograd.Function):
        @staticmethod
        # type: ignore[override]
        def forward(ctx: Any, grad_output: Tensor, input: Tensor) -> Tensor:
            ctx.save_for_backward(
                grad_output if input.requires_grad else NULL_TENSOR,
                input if grad_output.requires_grad else NULL_TENSOR,
            )
            ctx.grad_output_shape = grad_output.shape
            ctx.input_shape = input.shape
            grad_weight = weight_operation(
                input,
                weight_shape,
                grad_output,
                stride,
                padding,
                dilation,
                groups,
            )
            if grad_weight.shape != weight_shape:
                raise ValueError(
                    "weight shape mismatch in forward of GFixConvNdGradWeight"
                    f"(weight: {weight_shape}, grad_weight: {grad_weight.shape})"
                )
            return grad_weight

        @staticmethod
        # type: ignore[override]
        def backward(
            ctx: Any, second_grad_weight: Tensor
        ) -> Tuple[Optional[Tensor], ...]:
            grad_output, input = ctx.saved_tensors
            grad_output_shape = ctx.grad_output_shape
            input_shape = ctx.input_shape
            second_grad_output = None
            second_grad_input = None

            if ctx.needs_input_grad[0]:
                second_grad_output = GFixConvNd.apply(
                    input, second_grad_weight, None
                )
                assert second_grad_output.shape == grad_output_shape

            if ctx.needs_input_grad[1]:
                if transposed:
                    _output_padding = (0,) * rank
                else:
                    _output_padding = _calc_output_padding(
                        input_shape=grad_output_shape,
                        output_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=output_padding,
                        dilation=dilation,
                    )
                second_grad_input = _load_operation(
                    transposed=not transposed,
                    output_padding=_output_padding,
                    weight_shape=weight_shape,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    device=device,
                )(grad_output, second_grad_weight, None)
                if second_grad_input.shape != input_shape:
                    raise ValueError(
                        "second_grad_input shape mismatch in backward of GFixConvNdGradWeight"
                        f"(input: {input_shape}, second_grad_input: {second_grad_input.shape})"
                    )
            return second_grad_output, second_grad_input

    _CONV_CUSTOM_GRAD_CACHE[cache_key] = GFixConvNd
    return GFixConvNd.apply
