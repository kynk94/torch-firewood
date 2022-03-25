import itertools
import random
from typing import Type, Union

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from firewood.layers import conv_gradfix
from tests.helpers.runif import runif
from tests.helpers.utils import gen_params
from tests.stylegan3.torch_utils.ops import conv2d_gradfix


def test_conv_output_shape():
    lr = 1e-2
    C = 1
    padding_modes = ["zeros", 1, "reflect", "replicate", "circular"]
    for i in range(200):
        K = random.randint(1, 10)
        S = random.randint(1, 3)
        P_len = random.choice((2, 4))
        P = tuple(random.randint(0, 3) for _ in range(P_len))
        E = random.randint(S * K + sum(P), 100)
        O_H = (E - K + sum(P[: P_len // 2]) * 4 // P_len) // S + 1
        O_W = (E - K + sum(P[P_len // 2 :]) * 4 // P_len) // S + 1
        input_shape = (1, C, E, E)
        output_shape = (1, C, O_H, O_W)
        padding_mode = padding_modes[i % len(padding_modes)]

        input = torch.randn(input_shape)
        conv = conv_gradfix.GFixConv2d(
            C, C, K, S, P, bias=True, padding_mode=padding_mode
        )
        optimizer = torch.optim.Adam(conv.parameters(), lr=lr)
        optimizer.zero_grad()

        output: Tensor = conv(input)
        loss = F.mse_loss(output, torch.zeros_like(output))
        loss.backward()
        optimizer.step()

        assert (
            output.shape == output_shape
        ), "{} != {} for input: {}, kernel: {} stride: {} padding: {} padding_mode: {} reversed_padding: {}".format(
            tuple(output.shape),
            output_shape,
            input_shape,
            K,
            S,
            P,
            padding_mode,
            conv._reversed_padding_repeated_twice,
        )


def test_conv_transpose_output_shape():
    lr = 1e-2
    C = 1
    for i in range(200):
        K = random.randint(1, 10)
        S = random.randint(1, 3)
        P_len = random.choice((2, 4))
        P = tuple(random.randint(0, 3) for _ in range(P_len))
        E = random.randint(K + sum(P) + 10, 100)
        O = (E - 1) * S + K
        O_H = O - sum(P[: P_len // 2]) * 4 // P_len
        O_W = O - sum(P[P_len // 2 :]) * 4 // P_len
        input_shape = (1, C, E, E)
        output_shape = (1, C, O_H, O_W)

        input = torch.randn(input_shape)
        conv = conv_gradfix.GFixConvTranspose2d(C, C, K, S, P, bias=True)
        optimizer = torch.optim.Adam(conv.parameters(), lr=lr)
        optimizer.zero_grad()

        output: Tensor = conv(input)
        loss = F.mse_loss(output, torch.zeros_like(output))
        loss.backward()
        optimizer.step()

        assert (
            output.shape == output_shape
        ), "{} != {} for input: {}, kernel: {} stride: {} padding: {} reversed_padding: {}".format(
            tuple(output.shape),
            output_shape,
            input_shape,
            K,
            S,
            P,
            conv._reversed_padding_repeated_twice,
        )


@pytest.mark.parametrize(
    *gen_params(
        ["rank", "stride", "padding"],
        itertools.product([1, 2, 3], [1, 2], [0, "same"]),
    )
)
def test_no_weight_gradients_in_gfix_conv(
    rank: int, stride: int, padding: Union[str, int]
) -> None:
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank), requires_grad=True
    )
    x_copy = x.detach().requires_grad_()

    operation: Type[conv_gradfix._GFixConvNd] = getattr(
        conv_gradfix, f"GFixConv{rank}d"
    )
    conv = operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )
    conv_copy = operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )
    conv_copy.weight.data = conv.weight.data
    conv_copy.bias.data = conv.bias.data

    y: Tensor = conv(x)
    y_copy: Tensor = conv_copy(x_copy)

    with conv_gradfix.no_weight_gradients_in_gfix_conv():
        weight_grad = torch.autograd.grad(
            outputs=[y.square().sum() + conv.weight.square().sum()],
            inputs=[conv.weight],
            create_graph=True,
        )[0]
    weight_grad_copy = torch.autograd.grad(
        outputs=[y_copy.square().sum() + conv_copy.weight.square().sum()],
        inputs=[conv_copy.weight],
        create_graph=True,
    )[0]

    assert not torch.allclose(
        weight_grad, weight_grad_copy, atol=1e-1
    ), f"Forward weight_grad should be different. l1: {F.l1_loss(weight_grad, weight_grad_copy)}"


@pytest.mark.parametrize(*gen_params(["rank"], [1, 2, 3]))
def test_conv_same_padding_gradient_with_nn(rank: int) -> None:
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    stride = 2
    embedding_size = 7

    x = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank), requires_grad=True
    )
    x_copy = x.detach().requires_grad_()

    operation: Type[conv_gradfix._GFixConvNd] = getattr(
        conv_gradfix, f"GFixConv{rank}d"
    )
    conv = operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding="same",
        bias=True,
    )
    conv_copy = operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=2,
        bias=True,
    )
    conv_copy.weight.data = conv.weight.data
    conv_copy.bias.data = conv.bias.data

    y: Tensor = conv(x)
    y_copy: Tensor = conv_copy(x_copy)
    assert (
        y.shape == y_copy.shape
    ), f"Forward output shape mismatch. {y.shape} != {y_copy.shape}"
    assert torch.allclose(
        y, y_copy
    ), f"Forward result mismatch. l1: {F.l1_loss(y, y_copy)}"

    weight_grad = torch.autograd.grad(
        outputs=[y.square().sum() + conv.weight.square().sum()],
        inputs=[conv.weight],
        create_graph=True,
    )[0]
    weight_grad_copy = torch.autograd.grad(
        outputs=[y_copy.square().sum() + conv_copy.weight.square().sum()],
        inputs=[conv_copy.weight],
        create_graph=True,
    )[0]

    assert torch.allclose(
        weight_grad, weight_grad_copy
    ), f"Forward weight_grad mismatch. l1: {F.l1_loss(weight_grad, weight_grad_copy)}"


@pytest.mark.parametrize(
    *gen_params(
        ["rank", "stride", "padding"],
        itertools.product([1, 2, 3], [1, 2], [1, 2]),
    )
)
def test_conv_with_nn_cpu(rank: int, stride: int, padding: int) -> None:
    lr = 1e-2
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x_custom = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank), requires_grad=True
    )
    x_original = x_custom.detach().requires_grad_()

    custom_operation: Type[conv_gradfix._GFixConvNd] = getattr(
        conv_gradfix, f"GFixConv{rank}d"
    )
    nn_operation: Type[nn.modules.conv._ConvNd] = getattr(nn, f"Conv{rank}d")
    custom_conv = custom_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )
    nn_conv = nn_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )
    nn_conv.weight.data = custom_conv.weight.data
    nn_conv.bias.data = custom_conv.bias.data

    custom_parameters = [x_custom, *custom_conv.parameters()]
    original_parameters = [x_original, *nn_conv.parameters()]
    optimizer_custom = torch.optim.Adam(custom_parameters, lr=lr)
    optimizer_original = torch.optim.Adam(original_parameters, lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_conv(x_custom)
    y_original: Tensor = nn_conv(x_original)

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss_custom], inputs=[custom_conv.weight], create_graph=True
    )[0]
    weight_grad_original = torch.autograd.grad(
        outputs=[loss_original], inputs=[nn_conv.weight], create_graph=True
    )[0]

    absolute_tolerence = 1e-6 * 10**rank
    assert torch.allclose(
        y_custom, y_original
    ), f"Forward output mismatch. l1: {F.l1_loss(y_custom, y_original)}"
    assert torch.allclose(
        weight_grad_custom, weight_grad_original, atol=absolute_tolerence
    ), f"Forward weight_grad mismatch. l1: {F.l1_loss(weight_grad_custom, weight_grad_original)}"

    loss_custom += weight_grad_custom.square().sum()
    loss_original += weight_grad_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward input mismatch. l1: {F.l1_loss(x_custom, x_original)}"
    assert torch.allclose(
        custom_conv.weight, nn_conv.weight
    ), f"Backward weight mismatch. l1: {F.l1_loss(custom_conv.weight, nn_conv.weight)}"
    assert torch.allclose(
        custom_conv.bias, nn_conv.bias
    ), f"Backward bias mismatch. l1: {F.l1_loss(custom_conv.bias, nn_conv.bias)}"


@runif(min_gpus=1)
@pytest.mark.parametrize(
    *gen_params(
        ["rank", "stride", "padding"],
        itertools.product(range(1, 4), range(1, 3), range(2)),
    )
)
def test_conv_with_nn_gpu(rank: int, stride: int, padding: int) -> None:
    lr = 1e-2
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x_custom = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank),
        requires_grad=True,
        device="cuda",
    )
    x_original = x_custom.detach().requires_grad_()

    custom_operation: Type[conv_gradfix._GFixConvNd] = getattr(
        conv_gradfix, f"GFixConv{rank}d"
    )
    nn_operation: Type[nn.modules.conv._ConvNd] = getattr(nn, f"Conv{rank}d")
    custom_conv = custom_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    ).cuda()
    nn_conv = nn_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    ).cuda()
    nn_conv.weight.data = custom_conv.weight.data
    nn_conv.bias.data = custom_conv.bias.data

    custom_parameters = [x_custom, *custom_conv.parameters()]
    original_parameters = [x_original, *nn_conv.parameters()]
    optimizer_custom = torch.optim.Adam(custom_parameters, lr=lr)
    optimizer_original = torch.optim.Adam(original_parameters, lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_conv(x_custom)
    y_original: Tensor = nn_conv(x_original)

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss_custom], inputs=[custom_conv.weight], create_graph=True
    )[0]
    weight_grad_original = torch.autograd.grad(
        outputs=[loss_original], inputs=[nn_conv.weight], create_graph=True
    )[0]

    absolute_tolerence = 1e-7 * 10**rank
    assert torch.allclose(
        y_custom, y_original
    ), f"Forward output mismatch. l1: {F.l1_loss(y_custom, y_original)}"
    assert torch.allclose(
        weight_grad_custom, weight_grad_original, atol=absolute_tolerence
    ), f"Forward weight_grad mismatch. l1: {F.l1_loss(weight_grad_custom, weight_grad_original)}"

    loss_custom += weight_grad_custom.square().sum()
    loss_original += weight_grad_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward input mismatch. l1: {F.l1_loss(x_custom, x_original)}"
    assert torch.allclose(
        custom_conv.weight, nn_conv.weight
    ), f"Backward weight mismatch. l1: {F.l1_loss(custom_conv.weight, nn_conv.weight)}"
    assert torch.allclose(
        custom_conv.bias, nn_conv.bias
    ), f"Backward bias mismatch. l1: {F.l1_loss(custom_conv.bias, nn_conv.bias)}"


@pytest.mark.parametrize(
    *gen_params(
        ["rank", "stride", "padding"],
        itertools.product(range(1, 4), range(1, 3), range(2)),
    )
)
def test_conv_transpose_with_nn_cpu(
    rank: int, stride: int, padding: int
) -> None:
    lr = 1e-2
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x_custom = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank), requires_grad=True
    )
    x_original = x_custom.detach().requires_grad_()

    custom_operation: Type[conv_gradfix._GFixConvNd] = getattr(
        conv_gradfix, f"GFixConvTranspose{rank}d"
    )
    nn_operation: Type[nn.modules.conv._ConvTransposeNd] = getattr(
        nn, f"ConvTranspose{rank}d"
    )
    custom_conv = custom_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )
    nn_conv = nn_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )
    nn_conv.weight.data = custom_conv.weight.data
    nn_conv.bias.data = custom_conv.bias.data

    custom_parameters = [x_custom, *custom_conv.parameters()]
    original_parameters = [x_original, *nn_conv.parameters()]
    optimizer_custom = torch.optim.Adam(custom_parameters, lr=lr)
    optimizer_original = torch.optim.Adam(original_parameters, lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_conv(x_custom)
    y_original: Tensor = nn_conv(x_original)

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss_custom], inputs=[custom_conv.weight], create_graph=True
    )[0]
    weight_grad_original = torch.autograd.grad(
        outputs=[loss_original], inputs=[nn_conv.weight], create_graph=True
    )[0]

    absolute_tolerence = 1e-7 * 10**rank
    assert torch.allclose(
        y_custom, y_original, atol=absolute_tolerence
    ), f"Forward output mismatch. l1: {F.l1_loss(y_custom, y_original)}"
    assert torch.allclose(
        weight_grad_custom, weight_grad_original, atol=absolute_tolerence
    ), f"Forward weight_grad mismatch. l1: {F.l1_loss(weight_grad_custom, weight_grad_original)}"

    loss_custom += weight_grad_custom.square().sum()
    loss_original += weight_grad_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward input mismatch. l1: {F.l1_loss(x_custom, x_original)}"
    assert torch.allclose(
        custom_conv.weight, nn_conv.weight
    ), f"Backward weight mismatch. l1: {F.l1_loss(custom_conv.weight, nn_conv.weight)}"
    assert torch.allclose(
        custom_conv.bias, nn_conv.bias
    ), f"Backward bias mismatch. l1: {F.l1_loss(custom_conv.bias, nn_conv.bias)}"


@runif(min_gpus=1)
@pytest.mark.parametrize(
    *gen_params(
        ["rank", "stride", "padding"],
        itertools.product(range(1, 4), range(1, 3), range(2)),
    )
)
def test_conv_transpose_with_nn_gpu(
    rank: int, stride: int, padding: int
) -> None:
    lr = 1e-2
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x_custom = torch.randn(
        size=(2, in_channels, *(embedding_size,) * rank),
        requires_grad=True,
        device="cuda",
    )
    x_original = x_custom.detach().requires_grad_()

    custom_operation: Type[conv_gradfix._GFixConvNd] = getattr(
        conv_gradfix, f"GFixConvTranspose{rank}d"
    )
    nn_operation: Type[nn.modules.conv._ConvTransposeNd] = getattr(
        nn, f"ConvTranspose{rank}d"
    )
    custom_conv = custom_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    ).cuda()
    nn_conv = nn_operation(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    ).cuda()
    nn_conv.weight.data = custom_conv.weight.data
    nn_conv.bias.data = custom_conv.bias.data

    custom_parameters = [x_custom, *custom_conv.parameters()]
    original_parameters = [x_original, *nn_conv.parameters()]
    optimizer_custom = torch.optim.Adam(custom_parameters, lr=lr)
    optimizer_original = torch.optim.Adam(original_parameters, lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_conv(x_custom)
    y_original: Tensor = nn_conv(x_original)

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss_custom], inputs=[custom_conv.weight], create_graph=True
    )[0]
    weight_grad_original = torch.autograd.grad(
        outputs=[loss_original], inputs=[nn_conv.weight], create_graph=True
    )[0]

    absolute_tolerence = 5e-5 * 10 ** (rank - padding)
    assert torch.allclose(
        y_custom, y_original, rtol=1e-4, atol=absolute_tolerence
    ), f"Forward output mismatch. l1: {F.l1_loss(y_custom, y_original)}"
    assert torch.allclose(
        weight_grad_custom,
        weight_grad_original,
        rtol=1e-4,
        atol=absolute_tolerence,
    ), f"Forward weight_grad mismatch. l1: {F.l1_loss(weight_grad_custom, weight_grad_original)}"

    loss_custom += weight_grad_custom.square().sum()
    loss_original += weight_grad_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward input mismatch. l1: {F.l1_loss(x_custom, x_original)}"
    assert torch.allclose(
        custom_conv.weight, nn_conv.weight
    ), f"Backward weight mismatch. l1: {F.l1_loss(custom_conv.weight, nn_conv.weight)}"
    assert torch.allclose(
        custom_conv.bias, nn_conv.bias
    ), f"Backward bias mismatch. l1: {F.l1_loss(custom_conv.bias, nn_conv.bias)}"


@runif(min_gpus=1)
@pytest.mark.parametrize(
    *gen_params(["stride", "padding"], itertools.product(range(1, 3), range(2)))
)
def test_conv2d_with_stylegan_gpu(stride: int, padding: int) -> None:
    conv2d_gradfix.enabled = True

    lr = 1e-2
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x_custom = torch.randn(
        size=(2, in_channels, embedding_size, embedding_size),
        requires_grad=True,
        device="cuda",
    )
    x_original = x_custom.detach().requires_grad_()

    custom_conv = conv_gradfix.GFixConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    ).cuda()
    weight = nn.Parameter(custom_conv.weight.data)
    bias = nn.Parameter(custom_conv.bias.data)

    custom_parameters = [x_custom, *custom_conv.parameters()]
    original_parameters = [x_original, weight, bias]
    optimizer_custom = torch.optim.Adam(custom_parameters, lr=lr)
    optimizer_original = torch.optim.Adam(original_parameters, lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_conv(x_custom)
    y_original: Tensor = conv2d_gradfix.conv2d(
        input=x_original,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
    )

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss_custom], inputs=[custom_conv.weight], create_graph=True
    )[0]
    weight_grad_original = torch.autograd.grad(
        outputs=[loss_original], inputs=[weight], create_graph=True
    )[0]

    assert torch.allclose(
        y_custom, y_original
    ), f"Forward output mismatch. l1: {F.l1_loss(y_custom, y_original)}"
    assert torch.allclose(
        weight_grad_custom, weight_grad_original
    ), f"Forward weight_grad mismatch. l1: {F.l1_loss(weight_grad_custom, weight_grad_original)}"

    loss_custom += weight_grad_custom.square().sum()
    loss_original += weight_grad_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward input mismatch. l1: {F.l1_loss(x_custom, x_original)}"
    assert torch.allclose(
        custom_conv.weight, weight
    ), f"Backward weight mismatch. l1: {F.l1_loss(custom_conv.weight, weight)}"
    assert torch.allclose(
        custom_conv.bias, bias
    ), f"Backward bias mismatch. l1: {F.l1_loss(custom_conv.bias, bias)}"

    conv2d_gradfix.enabled = False


@runif(min_gpus=1)
@pytest.mark.parametrize(
    *gen_params(["stride", "padding"], itertools.product(range(1, 3), range(2)))
)
def test_conv_transpose2d_with_stylegan_gpu(stride: int, padding: int) -> None:
    conv2d_gradfix.enabled = True

    lr = 1e-2
    in_channels = 3
    out_channels = 10
    kernel_size = 5
    embedding_size = 10

    x_custom = torch.randn(
        size=(2, in_channels, embedding_size, embedding_size),
        requires_grad=True,
        device="cuda",
    )
    x_original = x_custom.detach().requires_grad_()

    custom_conv = conv_gradfix.GFixConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    ).cuda()
    weight = nn.Parameter(custom_conv.weight.data)
    bias = nn.Parameter(custom_conv.bias.data)

    custom_parameters = [x_custom, *custom_conv.parameters()]
    original_parameters = [x_original, weight, bias]
    optimizer_custom = torch.optim.Adam(custom_parameters, lr=lr)
    optimizer_original = torch.optim.Adam(original_parameters, lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_conv(x_custom)
    y_original: Tensor = conv2d_gradfix.conv_transpose2d(
        input=x_original,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
    )

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()

    weight_grad_custom = torch.autograd.grad(
        outputs=[loss_custom], inputs=[custom_conv.weight], create_graph=True
    )[0]
    weight_grad_original = torch.autograd.grad(
        outputs=[loss_original], inputs=[weight], create_graph=True
    )[0]

    assert torch.allclose(
        y_custom, y_original, atol=1e-5
    ), f"Forward output mismatch. l1: {F.l1_loss(y_custom, y_original)}"
    assert torch.allclose(
        weight_grad_custom, weight_grad_original, atol=1e-5
    ), f"Forward weight_grad mismatch. l1: {F.l1_loss(weight_grad_custom, weight_grad_original)}"

    loss_custom += weight_grad_custom.square().sum()
    loss_original += weight_grad_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward input mismatch. l1: {F.l1_loss(x_custom, x_original)}"
    assert torch.allclose(
        custom_conv.weight, weight
    ), f"Backward weight mismatch. l1: {F.l1_loss(custom_conv.weight, weight)}"
    assert torch.allclose(
        custom_conv.bias, bias
    ), f"Backward bias mismatch. l1: {F.l1_loss(custom_conv.bias, bias)}"

    conv2d_gradfix.enabled = False


@runif(tensorflow_installed=True)
@pytest.mark.parametrize(
    *gen_params(
        ["rank", "transposed", "stride"],
        itertools.product([1, 2, 3], [False, True], [1, 2]),
    )
)
def test_same_padding_with_tensorflow(
    rank: int, transposed: bool, stride: int
) -> None:
    import tensorflow as tf

    lr = 1e-2
    padding = "same"
    in_channels = 3
    out_channels = 3
    kernel_size = 4
    embedding_size = 19

    tf_input = tf.Variable(
        tf.random.normal((2, *(embedding_size,) * rank, in_channels)),
    )
    original_input = torch.tensor(tf_input.numpy()).permute(
        0, -1, *range(1, 1 + rank)
    )
    torch_input = original_input.clone().requires_grad_()

    tf_conv = getattr(
        tf.keras.layers,
        f"Conv{rank}DTranspose" if transposed else f"Conv{rank}D",
    )(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        data_format="channels_last",
        use_bias=True,
    )
    tf_conv.build(tf_input.shape)
    torch_conv: Type[conv_gradfix._GFixConvNd] = getattr(
        conv_gradfix,
        f"GFixConvTranspose{rank}d" if transposed else f"GFixConv{rank}d",
    )(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=True,
    )
    torch_conv.weight.data = torch.tensor(tf_conv.kernel.numpy()).permute(
        *range(rank + 1, rank - 1, -1), *range(rank)
    )
    torch_conv.bias.data = torch.tensor(tf_conv.bias.numpy())

    tf_optimizer = tf.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )
    torch_optimizer = torch.optim.Adam(
        [torch_input, *torch_conv.parameters()],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-7,
    )

    with tf.GradientTape() as tape:
        tf_output = tf_conv(tf_input)
        loss_tf = tf.reduce_sum(tf_output**2)
    gradient_tf = tape.gradient(
        loss_tf, [tf_input, *tf_conv.trainable_variables]
    )
    tf_optimizer.apply_gradients(
        zip(gradient_tf, [tf_input, *tf_conv.trainable_variables])
    )

    torch_optimizer.zero_grad()
    torch_output: Tensor = torch_conv(torch_input)
    loss_torch = torch_output.square().sum()
    loss_torch.backward()
    torch_optimizer.step()

    tf_input_torch = torch.tensor(tf_input.numpy()).permute(
        0, -1, *range(1, 1 + rank)
    )
    tf_output_torch = torch.tensor(tf_output.numpy()).permute(
        0, -1, *range(1, 1 + rank)
    )
    tf_weight_torch = torch.tensor(tf_conv.kernel.numpy()).permute(
        *range(rank + 1, rank - 1, -1), *range(rank)
    )

    assert torch.allclose(
        tf_output_torch, torch_output, rtol=1e-4, atol=1e-5
    ), f"Forward output mismatch. l1: {F.l1_loss(tf_output_torch, torch_output)}"
    assert torch.allclose(
        tf_weight_torch, torch_conv.weight, rtol=1e-4, atol=1e-5
    ), f"Backward weight mismatch. l1: {F.l1_loss(tf_weight_torch, torch_conv.weight)}"

    # If print inputs and compare it, it's almost equal.
    # However, the absolut tolerance value requires a high value.
    assert torch.allclose(
        tf_input_torch, torch_input, rtol=1e-4, atol=1e-2
    ), f"Backward input mismatch. l1: {F.l1_loss(tf_input_torch, torch_input)}"
    assert not torch.allclose(
        original_input, torch_input, rtol=1e-4, atol=1e-5
    ), f"Input value should be changed after backward. l1: {F.l1_loss(original_input, torch_input)}"
