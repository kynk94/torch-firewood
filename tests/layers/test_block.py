import itertools

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from firewood.layers.block import Block, normalize_op_order
from tests.helpers.raiseif import expect_raise


@pytest.mark.parametrize(
    "op_order", ("".join(i) for i in itertools.permutations("WNA", 3))
)
@torch.no_grad()
def test_op_order_with_nn(op_order):
    input = torch.randn(2, 3, 5, 5)
    weight_layer = nn.Conv2d(3, 3, 3, bias=False)
    normalization_layer = nn.BatchNorm2d(3)
    activation_layer = nn.ReLU()

    layer_dict = {
        "W": weight_layer.eval(),
        "N": normalization_layer.eval(),
        "A": activation_layer,
    }

    def nn_forward(input: Tensor) -> Tensor:
        output = input
        for order in op_order:
            output = layer_dict[order](output)
        return output

    block = Block(
        weight_layer=weight_layer,
        op_order=op_order,
        normalization="bn",
        activation="relu",
    ).eval()

    nn_output = nn_forward(input)
    block_output = block(input)
    assert torch.allclose(
        nn_output, block_output, atol=1e-7
    ), f"Forward result mismatch. l1: {F.l1_loss(nn_output, block_output)}"


def test_normalize_op_order():
    op_orders = tuple("".join(i) for i in itertools.permutations("WNA", 3))
    for op_order in op_orders:
        _op_order = op_order
        if "WN" not in op_order:
            _op_order = op_order.replace("W", "WB")
        assert normalize_op_order(op_order) == _op_order
    with expect_raise(ValueError):
        normalize_op_order("")
    with expect_raise(ValueError):
        normalize_op_order("WAA")
