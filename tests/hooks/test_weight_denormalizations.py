import pytest
import torch
import torch.nn.functional as F

from firewood.hooks.weight_denormalizations import remove_weight_denorm
from firewood.layers.conv_blocks import Conv2dBlock
from tests.helpers.runif import runif
from tests.stylegan3.training import networks_stylegan2, networks_stylegan3


def test_assign_and_remove() -> None:
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 5)

    kwargs = dict(
        bias=True,
        activation="lrelu",
        weight_normalization="denorm",
        weight_normalization_args={"modulation_features": 5},
    )
    block = Conv2dBlock(3, 3, 3, 2, 1, **kwargs)

    assert hasattr(block.layers["weighting"], "weight_orig")
    assert hasattr(block.layers["weighting"], "bias_orig")

    block(x, s)
    remove_weight_denorm(block)
    block(x)

    assert not hasattr(block.layers["weighting"], "weight_orig")
    assert not hasattr(block.layers["weighting"], "bias_orig")


@runif(min_gpus=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_with_stylegan2(dtype: torch.dtype) -> None:
    x = torch.randn(2, 3, 64, 64, dtype=dtype, device="cuda")
    s = torch.randn(2, 5, dtype=dtype, device="cuda")

    kwargs = dict(
        bias=False,
        weight_normalization="denorm",
        weight_normalization_args={"modulation_features": 5},
    )
    block = Conv2dBlock(3, 3, 3, **kwargs).cuda()
    weight = block.layers["weighting"].weight_orig.detach()

    output_custom = block(x, s)
    styles = block.layers["weighting"].gamma_affine(s)
    output_official = networks_stylegan2.modulated_conv2d(
        x, weight, styles, flip_weight=True
    )

    assert torch.allclose(
        output_custom, output_official, rtol=1e-4, atol=1e-7
    ), f"Forward result mismatch. l1: {F.l1_loss(output_custom, output_official)}"


def test_with_stylegan3() -> None:
    x = torch.randn(2, 3, 64, 64)
    s = torch.randn(2, 5)

    kwargs = dict(
        bias=False,
        weight_normalization="denorm",
        weight_normalization_args={
            "modulation_features": 5,
            "pre_normalize": "stylegan3",
        },
    )
    block = Conv2dBlock(3, 3, 3, **kwargs)
    weight = block.layers["weighting"].weight_orig.detach()

    output_custom = block(x, s)
    styles = block.layers["weighting"].gamma_affine(s)
    output_official = networks_stylegan3.modulated_conv2d(x, weight, styles)

    assert torch.allclose(
        output_custom, output_official, rtol=1e-4, atol=1e-7
    ), f"Forward result mismatch. l1: {F.l1_loss(output_custom, output_official)}"
