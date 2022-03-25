import random

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from firewood.layers.biased_activations import BiasedActivation
from tests.helpers.runif import runif
from tests.stylegan3.torch_utils.ops.bias_act import activation_funcs, bias_act


@pytest.mark.xfail(raises=ValueError)
def test_not_supported_activation_func():
    BiasedActivation(activation="invalid")


def test_bias_gain_clamp():
    bias_gain = 0.5
    biased_activation = BiasedActivation(
        activation="relu", gain=1.0, bias_gain=bias_gain, clamp=0.5
    )
    relu = nn.ReLU()

    input = torch.randn(2, 3, 5, 5)
    bias = torch.randn(3)
    biased_activation_output: Tensor = biased_activation(input, bias)
    relu_output: Tensor = relu(input + bias_gain * bias.view(1, -1, 1, 1))
    relu_output = relu_output.clamp_(-0.5, 0.5)
    assert torch.allclose(
        biased_activation_output, relu_output
    ), f"Forward result mismatch. l1: {F.l1_loss(biased_activation_output, relu_output)}"


@pytest.mark.parametrize("activation", activation_funcs)
def test_with_bias_cpu(activation: str) -> None:
    lr = 1e-2
    embedding_size = random.randint(1, 32)
    alpha = activation_funcs[activation]["def_alpha"]
    if activation == "elu":
        alpha = 1.0
    custom_operation = BiasedActivation(activation, alpha=alpha)

    x_custom = torch.randn(2, embedding_size, requires_grad=True)
    b_custom = torch.randn(embedding_size, requires_grad=True)
    x_original = x_custom.detach().requires_grad_()
    b_original = b_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom, b_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original, b_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_operation(x_custom, b_custom)
    y_original: Tensor = bias_act(
        x_original,
        b_original,
        act=activation,
        alpha=alpha,
        gain=custom_operation.gain,
        impl="ref",
    )
    assert torch.allclose(
        y_custom, y_original
    ), f"Forward result mismatch. l1: {F.l1_loss(y_custom, y_original)}"

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward input mismatch. l1: {F.l1_loss(x_custom, x_original)}"
    assert torch.allclose(
        b_custom, b_original
    ), f"Backward bias mismatch. l1: {F.l1_loss(b_custom, b_original)}"


@pytest.mark.parametrize("activation", activation_funcs)
def test_without_bias_cpu(activation: str) -> None:
    lr = 1e-2
    embedding_size = random.randint(1, 32)
    alpha = activation_funcs[activation]["def_alpha"]
    if activation == "elu":
        alpha = 1.0
    custom_operation = BiasedActivation(activation, alpha=alpha)

    x_custom = torch.randn(2, embedding_size, requires_grad=True)
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_operation(x_custom)
    y_original: Tensor = bias_act(
        x_original,
        act=activation,
        alpha=alpha,
        gain=custom_operation.gain,
        impl="ref",
    )
    assert torch.allclose(
        y_custom, y_original
    ), f"Forward result mismatch. l1: {F.l1_loss(y_custom, y_original)}"

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward result mismatch. l1: {F.l1_loss(x_custom, x_original)}"


@runif(min_gpus=1)
@pytest.mark.parametrize("activation", activation_funcs)
def test_with_bias_gpu(activation: str) -> None:
    lr = 1e-2
    embedding_size = random.randint(1, 32)
    alpha = activation_funcs[activation]["def_alpha"]
    custom_operation = BiasedActivation(activation, alpha=alpha).cuda()

    x_custom = torch.randn(2, embedding_size, requires_grad=True, device="cuda")
    b_custom = torch.randn(embedding_size, requires_grad=True, device="cuda")
    x_original = x_custom.detach().requires_grad_()
    b_original = b_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom, b_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original, b_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_operation(x_custom, b_custom)
    y_original: Tensor = bias_act(
        x_original,
        b_original,
        act=activation,
        alpha=alpha,
        gain=custom_operation.gain,
        impl="cuda",
    )
    assert torch.allclose(
        y_custom, y_original
    ), f"Forward result mismatch. l1: {F.l1_loss(y_custom, y_original)}"

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward input mismatch. l1: {F.l1_loss(x_custom, x_original)}"
    assert torch.allclose(
        b_custom, b_original
    ), f"Backward bias mismatch. l1: {F.l1_loss(b_custom, b_original)}"


@runif(min_gpus=1)
@pytest.mark.parametrize("activation", activation_funcs)
def test_without_bias_gpu(activation: str) -> None:
    lr = 1e-2
    embedding_size = random.randint(1, 32)
    alpha = activation_funcs[activation]["def_alpha"]
    custom_operation = BiasedActivation(activation, alpha=alpha).cuda()

    x_custom = torch.randn(2, embedding_size, requires_grad=True, device="cuda")
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    y_custom: Tensor = custom_operation(x_custom)
    y_original: Tensor = bias_act(
        x_original,
        act=activation,
        alpha=alpha,
        gain=custom_operation.gain,
        impl="cuda",
    )
    assert torch.allclose(
        y_custom, y_original
    ), f"Forward result mismatch. l1: {F.l1_loss(y_custom, y_original)}"

    loss_custom = y_custom.square().sum()
    loss_original = y_original.square().sum()
    loss_custom.backward()
    loss_original.backward()

    optimizer_custom.step()
    optimizer_original.step()

    assert torch.allclose(
        x_custom, x_original
    ), f"Backward result mismatch. l1: {F.l1_loss(x_custom, x_original)}"
