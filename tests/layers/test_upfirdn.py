import itertools
import random

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from firewood.layers.upfirdn import UpFirDn2d
from tests.helpers.raiseif import raiseif
from tests.helpers.runif import runif
from tests.helpers.utils import gen_params, power_of_2
from tests.stylegan3.torch_utils.ops.upfirdn2d import (
    downsample2d,
    filter2d,
    setup_filter,
    upfirdn2d,
    upsample2d,
)


@pytest.mark.parametrize(*gen_params("padding", range(-4, 5, 2)))
def test_filter2d_cpu(padding: int) -> None:
    lr = 1e-2
    kernel = [random.randint(1, 5) for _ in range(4)]
    embedding_size = random.choice(tuple(power_of_2(6))[1:])
    custom_operation = UpFirDn2d(kernel, padding=padding)

    x_custom = torch.randn(
        size=(2, 3, embedding_size, embedding_size),
        requires_grad=True,
    )
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    output_size = (
        embedding_size + sum(custom_operation.padding) / 2 - len(kernel) + 1
    )
    with raiseif(output_size < 1, RuntimeError) as exception:
        y_custom: Tensor = custom_operation(x_custom)
        y_original: Tensor = filter2d(
            x=x_original,
            f=setup_filter(kernel),
            padding=padding,
            impl="ref",
        )
    if exception is not None:
        return
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
@pytest.mark.parametrize(*gen_params("padding", range(-4, 5, 2)))
def test_filter2d_gpu(padding: int) -> None:
    lr = 1e-2
    kernel = [random.randint(1, 5) for _ in range(4)]
    embedding_size = random.choice(tuple(power_of_2(6)))
    custom_operation = UpFirDn2d(kernel, padding=padding).cuda()

    x_custom = torch.randn(
        size=(2, 3, embedding_size, embedding_size),
        requires_grad=True,
        device="cuda",
    )
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    output_size = (
        embedding_size + sum(custom_operation.padding) / 2 - len(kernel) + 1
    )
    with raiseif(output_size < 1, RuntimeError) as exception:
        y_custom: Tensor = custom_operation(x_custom)
        y_original: Tensor = filter2d(
            x=x_original,
            f=setup_filter(kernel, device="cuda"),
            padding=padding,
            impl="cuda",
        )
    if exception is not None:
        return
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


@pytest.mark.parametrize(
    *gen_params(
        ["up", "padding"],
        itertools.product((2, 3), range(-2, 3)),
    )
)
def test_upfir2d_cpu(up: int, padding: int) -> None:
    lr = 1e-2
    kernel = [random.randint(1, 5) for _ in range(4)]
    embedding_size = random.choice(tuple(power_of_2(6)))
    custom_operation = UpFirDn2d(kernel, up, padding=padding)

    x_custom = torch.randn(
        size=(2, 3, embedding_size, embedding_size),
        requires_grad=True,
    )
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    output_size = (
        embedding_size * up
        + sum(custom_operation.padding) / 2
        - len(kernel)
        + 1
    )
    with raiseif(output_size < 1, RuntimeError) as exception:
        y_custom: Tensor = custom_operation(x_custom)
        y_original: Tensor = upsample2d(
            x=x_original,
            f=setup_filter(kernel),
            up=up,
            padding=padding,
            impl="ref",
        )
    if exception is not None:
        return
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
@pytest.mark.parametrize(
    *gen_params(
        ["up", "padding"],
        itertools.product((2, 3), range(-2, 3)),
    )
)
def test_upfir2d_gpu(up: int, padding: int) -> None:
    lr = 1e-2
    kernel = [random.randint(1, 5) for _ in range(4)]
    embedding_size = random.choice(tuple(power_of_2(6)))
    custom_operation = UpFirDn2d(kernel, up, padding=padding).cuda()

    x_custom = torch.randn(
        size=(2, 3, embedding_size, embedding_size),
        requires_grad=True,
        device="cuda",
    )
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    output_size = (
        embedding_size * up
        + sum(custom_operation.padding) / 2
        - len(kernel)
        + 1
    )
    with raiseif(output_size < 1, RuntimeError) as exception:
        y_custom: Tensor = custom_operation(x_custom)
        y_original: Tensor = upsample2d(
            x=x_original,
            f=setup_filter(kernel, device="cuda"),
            up=up,
            padding=padding,
            impl="cuda",
        )
    if exception is not None:
        return
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


@pytest.mark.parametrize(
    *gen_params(
        ["down", "padding"],
        itertools.product((2, 3), range(-2, 3)),
    )
)
def test_firdown2d_cpu(down: int, padding: int) -> None:
    lr = 1e-2
    kernel = [random.randint(1, 5) for _ in range(4)]
    embedding_size = random.choice(tuple(power_of_2(6)))
    custom_operation = UpFirDn2d(kernel, down=down, padding=padding)

    x_custom = torch.randn(
        size=(2, 3, embedding_size, embedding_size),
        requires_grad=True,
    )
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    output_size = (
        embedding_size + sum(custom_operation.padding) / 2 - len(kernel) + 1
    )
    with raiseif(output_size < 1, RuntimeError) as exception:
        y_custom: Tensor = custom_operation(x_custom)
        y_original: Tensor = downsample2d(
            x=x_original,
            f=setup_filter(kernel),
            down=down,
            padding=padding,
            impl="ref",
        )
    if exception is not None:
        return
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
@pytest.mark.parametrize(
    *gen_params(
        ["down", "padding"],
        itertools.product((2, 3), range(-2, 3)),
    )
)
def test_firdown2d_gpu(down: int, padding: int) -> None:
    lr = 1e-2
    kernel = [random.randint(1, 5) for _ in range(4)]
    embedding_size = random.choice(tuple(power_of_2(6)))
    custom_operation = UpFirDn2d(kernel, down=down, padding=padding).cuda()

    x_custom = torch.randn(
        size=(2, 3, embedding_size, embedding_size),
        requires_grad=True,
        device="cuda",
    )
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    output_size = (
        embedding_size + sum(custom_operation.padding) / 2 - len(kernel) + 1
    )
    with raiseif(output_size < 1, RuntimeError) as exception:
        y_custom: Tensor = custom_operation(x_custom)
        y_original: Tensor = downsample2d(
            x=x_original,
            f=setup_filter(kernel, device="cuda"),
            down=down,
            padding=padding,
            impl="cuda",
        )
    if exception is not None:
        return
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


@pytest.mark.parametrize(
    *gen_params(
        ["up", "down", "padding"],
        itertools.product((2, 3), (2, 3), range(-2, 3)),
    )
)
def test_upfirdn2d_cpu(up: int, down: int, padding: int) -> None:
    lr = 1e-2
    kernel = [random.randint(1, 5) for _ in range(4)]
    embedding_size = random.choice(tuple(power_of_2(6)))
    custom_operation = UpFirDn2d(kernel, up, down, padding)

    x_custom = torch.randn(
        size=(2, 3, embedding_size, embedding_size),
        requires_grad=True,
    )
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    output_size = (
        embedding_size * up
        + sum(custom_operation.padding) / 2
        - len(kernel)
        + 1
    )
    with raiseif(output_size < 1, RuntimeError) as exception:
        y_custom: Tensor = custom_operation(x_custom)
        y_original: Tensor = upfirdn2d(
            x=x_original,
            f=setup_filter(kernel),
            up=up,
            down=down,
            padding=custom_operation.padding,
            gain=up**2,
            impl="ref",
        )
    if exception is not None:
        return
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
@pytest.mark.parametrize(
    *gen_params(
        ["up", "down", "padding"],
        itertools.product((2, 3), (2, 3), range(-2, 3)),
    )
)
def test_upfirdn2d_gpu(up: int, down: int, padding: int) -> None:
    lr = 1e-2
    kernel = [random.randint(1, 5) for _ in range(4)]
    embedding_size = random.choice(tuple(power_of_2(6)))
    custom_operation = UpFirDn2d(kernel, up, down, padding).cuda()

    x_custom = torch.randn(
        size=(2, 3, embedding_size, embedding_size),
        requires_grad=True,
        device="cuda",
    )
    x_original = x_custom.detach().requires_grad_()

    optimizer_custom = torch.optim.Adam([x_custom], lr=lr)
    optimizer_original = torch.optim.Adam([x_original], lr=lr)
    optimizer_custom.zero_grad()
    optimizer_original.zero_grad()

    output_size = (
        embedding_size * up
        + sum(custom_operation.padding) / 2
        - len(kernel)
        + 1
    )
    with raiseif(output_size < 1, RuntimeError) as exception:
        y_custom: Tensor = custom_operation(x_custom)
        y_original: Tensor = upfirdn2d(
            x_original,
            setup_filter(kernel, device="cuda"),
            up=up,
            down=down,
            padding=custom_operation.padding,
            gain=up**2,
            impl="cuda",
        )
    if exception is not None:
        return
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
