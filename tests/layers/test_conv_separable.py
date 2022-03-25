import itertools
from typing import List, Type

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from firewood.layers import conv_separable
from tests.helpers.runif import runif
from tests.helpers.utils import gen_params


@pytest.mark.parametrize(
    *gen_params(
        ["rank", "transposed", "stride", "padding", "dilation"],
        itertools.product([1, 2, 3], [False, True], [1, 2], [0, 1], [1, 2]),
    )
)
def test_depth_separable_conv_shape(
    rank: int, transposed: bool, stride: int, padding: int, dilation: int
) -> None:
    in_channels = 3
    hidden_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank), requires_grad=True
    )

    operation: Type[conv_separable._SepConvNd] = getattr(
        conv_separable,
        f"DepthSepConvTranspose{rank}d"
        if transposed
        else f"DepthSepConv{rank}d",
    )
    nn_operation: Type[nn.modules.conv._ConvNd] = getattr(
        nn, f"ConvTranspose{rank}d" if transposed else f"Conv{rank}d"
    )
    conv = operation(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )
    nn_conv = nn_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )

    assert (
        conv.bias.shape == nn_conv.bias.shape
    ), f"Bias shape mismatch. separable: {conv.bias.shape}, nn: {nn_conv.bias.shape}"

    output: Tensor = conv(x)
    nn_output: Tensor = nn_conv(x)
    assert (
        output.shape == nn_output.shape
    ), f"Output shape mismatch. separable: {output.shape}, nn: {nn_output.shape}"


@pytest.mark.parametrize(
    *gen_params(
        ["rank", "transposed"], itertools.product([1, 2, 3], [False, True])
    )
)
def test_depth_separable_conv_backprop_cpu(rank: int, transposed: bool) -> None:
    lr = 1e-2
    in_channels = 3
    hidden_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank), requires_grad=True
    )

    operation: Type[conv_separable._SepConvNd] = getattr(
        conv_separable,
        f"DepthSepConvTranspose{rank}d"
        if transposed
        else f"DepthSepConv{rank}d",
    )
    conv = operation(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    )
    parameters: List[Tensor] = [x, *conv.parameters()]
    kept_parameters = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    optimizer.zero_grad()

    output: Tensor = conv(x)
    loss = output.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss],
        inputs=[conv.depthwise.weight, conv.pointwise.weight],
        create_graph=True,
    )[0]
    loss += weight_grad_custom.square().sum()

    loss.backward()
    optimizer.step()

    assert all(
        F.l1_loss(parameter, kept_parameter).item() > 1e-5
        for parameter, kept_parameter in zip(parameters, kept_parameters)
    ), "Parameters not updated correctly"


@runif(min_gpus=1)
@pytest.mark.parametrize(
    *gen_params(
        ["rank", "transposed"], itertools.product([1, 2, 3], [False, True])
    )
)
def test_depth_separable_conv_backprop_gpu(rank: int, transposed: bool) -> None:
    lr = 1e-2
    in_channels = 3
    hidden_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank),
        requires_grad=True,
        device="cuda",
    )

    operation: Type[conv_separable._SepConvNd] = getattr(
        conv_separable,
        f"DepthSepConvTranspose{rank}d"
        if transposed
        else f"DepthSepConv{rank}d",
    )
    conv = operation(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ).cuda()
    parameters: List[Tensor] = [x, *conv.parameters()]
    kept_parameters = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    optimizer.zero_grad()

    output: Tensor = conv(x)
    loss = output.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss],
        inputs=[conv.depthwise.weight, conv.pointwise.weight],
        create_graph=True,
    )[0]
    loss += weight_grad_custom.square().sum()

    loss.backward()
    optimizer.step()

    assert all(
        F.l1_loss(parameter, kept_parameter).item() > 1e-5
        for parameter, kept_parameter in zip(parameters, kept_parameters)
    ), "Parameters not updated correctly"


@pytest.mark.parametrize(
    *gen_params(
        ["rank", "transposed", "stride", "padding"],
        itertools.product([2, 3], [False, True], [1, 2], [0, 1]),
    )
)
def test_spatial_separable_conv_shape(
    rank: int, transposed: bool, stride: int, padding: int
) -> None:
    in_channels = 3
    out_channels = 3
    kernel_size = 3
    embedding_size = 10

    x = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank), requires_grad=True
    )

    operation: Type[conv_separable._SepConvNd] = getattr(
        conv_separable,
        f"SpatialSepConvTranspose{rank}d"
        if transposed
        else f"SpatialSepConv{rank}d",
    )
    nn_operation: Type[nn.modules.conv._ConvNd] = getattr(
        nn, f"ConvTranspose{rank}d" if transposed else f"Conv{rank}d"
    )
    conv = operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        bias=True,
    )
    nn_conv = nn_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        bias=True,
    )

    assert (
        conv.bias.shape == nn_conv.bias.shape
    ), f"Bias shape mismatch. separable: {conv.bias.shape}, nn: {nn_conv.bias.shape}"

    output: Tensor = conv(x)
    nn_output: Tensor = nn_conv(x)
    assert (
        output.shape == nn_output.shape
    ), f"Output shape mismatch. separable: {output.shape}, nn: {nn_output.shape}"


@pytest.mark.parametrize(
    *gen_params(
        ["rank", "transposed"], itertools.product([2, 3], [False, True])
    )
)
def test_spatial_separable_conv_backprop_cpu(
    rank: int, transposed: bool
) -> None:
    lr = 1e-2
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank), requires_grad=True
    )

    operation: Type[conv_separable._SepConvNd] = getattr(
        conv_separable,
        f"SpatialSepConvTranspose{rank}d"
        if transposed
        else f"SpatialSepConv{rank}d",
    )
    conv = operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    )
    parameters: List[Tensor] = [x, *conv.parameters()]
    kept_parameters = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    optimizer.zero_grad()

    output: Tensor = conv(x)
    loss = output.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss],
        inputs=[spatialwise.weight for spatialwise in conv.spatialwise],
        create_graph=True,
    )[0]
    loss += weight_grad_custom.square().sum()

    loss.backward()
    optimizer.step()

    assert all(
        F.l1_loss(parameter, kept_parameter).item() > 1e-5
        for parameter, kept_parameter in zip(parameters, kept_parameters)
    ), "Parameters not updated correctly"


@runif(min_gpus=1)
@pytest.mark.parametrize(
    *gen_params(
        ["rank", "transposed"], itertools.product([2, 3], [False, True])
    )
)
def test_spatial_separable_conv_backprop_gpu(
    rank: int, transposed: bool
) -> None:
    lr = 1e-2
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank),
        requires_grad=True,
        device="cuda",
    )

    operation: Type[conv_separable._SepConvNd] = getattr(
        conv_separable,
        f"SpatialSepConvTranspose{rank}d"
        if transposed
        else f"SpatialSepConv{rank}d",
    )
    conv = operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ).cuda()
    parameters: List[Tensor] = [x, *conv.parameters()]
    kept_parameters = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=lr)
    optimizer.zero_grad()

    output: Tensor = conv(x)
    loss = output.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss],
        inputs=[spatialwise.weight for spatialwise in conv.spatialwise],
        create_graph=True,
    )[0]
    loss += weight_grad_custom.square().sum()

    loss.backward()
    optimizer.step()

    assert all(
        F.l1_loss(parameter, kept_parameter).item() > 1e-5
        for parameter, kept_parameter in zip(parameters, kept_parameters)
    ), "Parameters not updated correctly"
