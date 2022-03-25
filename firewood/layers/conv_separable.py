import math
from typing import Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.utils import _pair, _single, _triple

from firewood.common.types import INT, SAME_PADDING
from firewood.layers import initializers
from firewood.layers.conv_gradfix import (
    GFixConv1d,
    GFixConv2d,
    GFixConv3d,
    GFixConvTranspose1d,
    GFixConvTranspose2d,
    GFixConvTranspose3d,
)
from firewood.utils import _pair_padding, _single_padding, _triple_padding


class _SepConvNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        bias: bool,
        bias_initializer: str = "zeros",
    ):
        super().__init__()
        self.rank = len(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias_initializer = bias_initializer

        if bias:
            self.bias = Parameter(torch.empty(self.out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.bias is None:
            return

        # bias initialization
        # add other init methods here if needed
        if self.bias_initializer == "uniform":  # default torch init
            fan_in = self.in_channels * math.prod(self.kernel_size)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            initializers.get(self.bias_initializer)(self.bias)

    def _add_bias(self, input: Tensor) -> Tensor:
        if self.bias is None:
            return input
        bias = self.bias.view([-1 if i == 1 else 1 for i in range(input.ndim)])
        return input + bias.to(dtype=input.dtype)


class _DepthSepConvNd(_SepConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Union[SAME_PADDING, INT],
        dilation: Tuple[int, ...],
        hidden_channels: int,
        bias: bool,
        padding_mode: str,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            bias_initializer=bias_initializer,
        )
        if self.rank == 1:
            convolution = GFixConv1d
        elif self.rank == 2:
            convolution = GFixConv2d  # type: ignore
        elif self.rank == 3:
            convolution = GFixConv3d  # type: ignore
        else:
            raise ValueError(f"Invalid kernel_size: {kernel_size}")
        self.hidden_channels = hidden_channels

        self.depthwise = convolution(
            in_channels=self.in_channels,
            out_channels=self.in_channels * self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
            padding_mode=padding_mode,
            weight_initializer=weight_initializer,
        )
        self.pointwise = convolution(
            in_channels=self.in_channels * self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            weight_initializer=weight_initializer,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._add_bias(self.pointwise(self.depthwise(input)))

    def extra_repr(self) -> str:
        return super().extra_repr() + ", ".join(
            [
                f"in_channels={self.in_channels}",
                f"out_channels={self.out_channels}",
                f"hidden_channels={self.hidden_channels}",
            ]
        )


class DepthSepConv1d(_DepthSepConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
        hidden_channels: int = 1,
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
            hidden_channels=hidden_channels,
            bias=bias,
            padding_mode=padding_mode,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class DepthSepConv2d(_DepthSepConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
        hidden_channels: int = 1,
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
            hidden_channels=hidden_channels,
            bias=bias,
            padding_mode=padding_mode,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class DepthSepConv3d(_DepthSepConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
        hidden_channels: int = 1,
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
            hidden_channels=hidden_channels,
            bias=bias,
            padding_mode=padding_mode,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class _DepthSepConvTransposeNd(_SepConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Union[SAME_PADDING, INT],
        dilation: Tuple[int, ...],
        output_padding: Tuple[int, ...],
        hidden_channels: int,
        bias: bool,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            bias_initializer=bias_initializer,
        )
        if self.rank == 1:
            convolution = GFixConvTranspose1d
        elif self.rank == 2:
            convolution = GFixConvTranspose2d  # type: ignore
        elif self.rank == 3:
            convolution = GFixConvTranspose3d  # type: ignore
        else:
            raise ValueError(f"Invalid kernel_size: {kernel_size}")
        self.hidden_channels = hidden_channels

        self.depthwise = convolution(
            in_channels=self.in_channels,
            out_channels=self.in_channels * self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            groups=in_channels,
            bias=False,
            weight_initializer=weight_initializer,
        )
        self.pointwise = convolution(
            in_channels=self.in_channels * self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            output_padding=0,
            groups=1,
            bias=False,
            weight_initializer=weight_initializer,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._add_bias(self.pointwise(self.depthwise(input)))

    def extra_repr(self) -> str:
        return super().extra_repr() + ", ".join(
            [
                f"in_channels={self.in_channels}",
                f"out_channels={self.out_channels}",
                f"hidden_channels={self.hidden_channels}",
            ]
        )


class DepthSepConvTranspose1d(_DepthSepConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
        output_padding: INT = 0,
        hidden_channels: int = 1,
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
            output_padding=_single(output_padding),
            hidden_channels=hidden_channels,
            bias=bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class DepthSepConvTranspose2d(_DepthSepConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
        output_padding: INT = 0,
        hidden_channels: int = 1,
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
            output_padding=_pair(output_padding),
            hidden_channels=hidden_channels,
            bias=bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class DepthSepConvTranspose3d(_DepthSepConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
        output_padding: INT = 0,
        hidden_channels: int = 1,
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
            output_padding=_triple(output_padding),
            hidden_channels=hidden_channels,
            bias=bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class _SpatialSepConvNd(_SepConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Union[SAME_PADDING, INT],
        dilation: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            bias_initializer=bias_initializer,
        )
        if self.rank == 2:
            convolution = GFixConv2d
        elif self.rank == 3:
            convolution = GFixConv3d  # type: ignore
        else:
            raise ValueError(f"Invalid kernel_size: {kernel_size}")
        n_parameters = in_channels * out_channels * math.prod(kernel_size)
        smaller = min(in_channels, out_channels)
        n_sep_parameters = smaller**2 * sum(kernel_size[:-1])
        n_sep_parameters += in_channels * out_channels * kernel_size[-1]
        if n_parameters < n_sep_parameters:
            raise ValueError(
                "There is no advantage to using spatial separable convolution. "
                f"Original: {n_parameters}, Separable: {n_sep_parameters}"
            )

        if isinstance(padding, (str, int)):
            _padding: Any = (padding,) * self.rank
        elif isinstance(padding, (tuple, list)):
            if len(padding) == self.rank:
                _padding = tuple(
                    (0,) * i + (p,) + (0,) * (self.rank - i - 1)
                    for i, p in enumerate(padding)
                )
            else:
                _padding = tuple(
                    (0, 0) * i
                    + tuple(padding[2 * i : 2 * (i + 1)])
                    + (0, 0) * (self.rank - i - 1)
                    for i in range(self.rank)
                )
        else:
            raise ValueError(f"Invalid padding: {padding}")

        in_C = (in_channels,) + (smaller,) * (self.rank - 1)
        out_C = (smaller,) * (self.rank - 1) + (out_channels,)

        self.spatialwise = nn.ModuleList()
        for i, (k, s) in enumerate(zip(kernel_size, stride)):
            _kernel_size = (1,) * i + (k,) + (1,) * (self.rank - i - 1)
            _stride = (1,) * i + (s,) + (1,) * (self.rank - i - 1)
            if i == 0:
                out_channels = in_channels
            elif i == self.rank - 1:
                out_channels = self.out_channels
            self.spatialwise.append(
                convolution(
                    in_channels=in_C[i],
                    out_channels=out_C[i],
                    kernel_size=_kernel_size,
                    stride=_stride,
                    padding=_padding[i],
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                    padding_mode=padding_mode,
                    weight_initializer=weight_initializer,
                )
            )

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for spatialwise in self.spatialwise:
            output = spatialwise(output)
        return self._add_bias(output)


class SpatialSepConv2d(_SpatialSepConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
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
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class SpatialSepConv3d(_SpatialSepConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
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
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class _SpatialSepConvTransposeNd(_SepConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Union[SAME_PADDING, INT],
        dilation: Tuple[int, ...],
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            bias_initializer=bias_initializer,
        )
        if self.rank == 2:
            convolution = GFixConvTranspose2d
        elif self.rank == 3:
            convolution = GFixConvTranspose3d  # type: ignore
        else:
            raise ValueError(f"Invalid kernel_size: {kernel_size}")
        n_parameters = in_channels * out_channels * math.prod(kernel_size)
        smaller = min(in_channels, out_channels)
        n_sep_parameters = smaller**2 * sum(kernel_size[:-1])
        n_sep_parameters += in_channels * out_channels * kernel_size[-1]
        if n_parameters < n_sep_parameters:
            raise ValueError(
                "There is no advantage to using spatial separable convolution. "
                f"Original: {n_parameters}, Separable: {n_sep_parameters}"
            )

        if isinstance(padding, (str, int)):
            _padding: Any = (padding,) * self.rank
        elif isinstance(padding, (tuple, list)):
            if len(padding) == self.rank:
                _padding = tuple(
                    (0,) * i + (p,) + (0,) * (self.rank - i - 1)
                    for i, p in enumerate(padding)
                )
            else:
                _padding = tuple(
                    (0, 0) * i
                    + tuple(padding[2 * i : 2 * (i + 1)])
                    + (0, 0) * (self.rank - i - 1)
                    for i in range(self.rank)
                )
        else:
            raise ValueError(f"Invalid padding: {padding}")

        in_C = (in_channels,) + (smaller,) * (self.rank - 1)
        out_C = (smaller,) * (self.rank - 1) + (out_channels,)

        self.spatialwise = nn.ModuleList()
        for i, (k, s) in enumerate(zip(kernel_size, stride)):
            _kernel_size = (1,) * i + (k,) + (1,) * (self.rank - i - 1)
            _stride = (1,) * i + (s,) + (1,) * (self.rank - i - 1)
            self.spatialwise.append(
                convolution(
                    in_channels=in_C[i],
                    out_channels=out_C[i],
                    kernel_size=_kernel_size,
                    stride=_stride,
                    padding=_padding[i],
                    dilation=dilation,
                    output_padding=0 if i < self.rank - 1 else output_padding,
                    groups=groups,
                    bias=False,
                    weight_initializer=weight_initializer,
                )
            )

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for spatialwise in self.spatialwise:
            output = spatialwise(output)
        return self._add_bias(output)


class SpatialSepConvTranspose2d(_SpatialSepConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
        output_padding: INT = 0,
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
            output_padding=_pair(output_padding),
            groups=groups,
            bias=bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )


class SpatialSepConvTranspose3d(_SpatialSepConvTransposeNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: INT,
        stride: INT = 1,
        padding: Union[SAME_PADDING, INT] = 0,
        dilation: INT = 1,
        output_padding: INT = 0,
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
            output_padding=_triple(output_padding),
            groups=groups,
            bias=bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )
