import itertools

import torch
import torch.nn as nn
from torch import Tensor

from firewood.utils import torch_op
from tests.helpers.raiseif import expect_raise


def test_is_cuda():
    assert torch_op.is_cuda("cuda")
    assert torch_op.is_cuda(0)
    assert torch_op.is_cuda(torch.device("cuda"))
    assert torch_op.is_cuda(torch.device("cuda:0"))
    assert not torch_op.is_cuda("cpu")
    assert not torch_op.is_cuda(torch.device("cpu"))
    assert not torch_op.is_cuda(None)


def test_normalize_activation_name():
    assert torch_op.normalize_activation_name("relu") == "relu"
    assert torch_op.normalize_activation_name("lrelu") == "leaky_relu"
    assert torch_op.normalize_activation_name("swish") == "silu"
    assert torch_op.normalize_activation_name(None) == "linear"
    others = ["linear", "sigmoid", "tanh", "softmax"]
    for other in others:
        assert torch_op.normalize_activation_name(other) == other


def test_normalize_op_order():
    op_orders = tuple("".join(i) for i in itertools.permutations("WNA", 3))
    for op_order in op_orders:
        assert torch_op.normalize_op_order(op_order) == op_order
    with expect_raise(ValueError):
        torch_op.normalize_op_order("")
    with expect_raise(ValueError):
        torch_op.normalize_op_order("WAA")


def test_get_in_out_features():
    in_features = 1
    out_features = 3
    kernel_size = 5
    layer = nn.Conv2d(in_features, out_features, kernel_size, bias=False)
    assert torch_op.get_in_out_features(layer) == (in_features, out_features)

    class TestModule(nn.Module):
        def __init__(
            self, in_features: int, out_features: int, kernel_size: int
        ) -> None:
            super().__init__()
            self.in_ = in_features
            self.out_ = out_features
            self.weight = nn.Parameter(
                torch.randn(out_features, in_features, kernel_size, kernel_size)
            )
            self.dummy = nn.Parameter(torch.randn(out_features))

        def forward(input: Tensor) -> Tensor:
            return input

    test_module = TestModule(in_features, out_features, kernel_size)
    assert torch_op.get_in_out_features(test_module) == (
        in_features,
        out_features,
    )
    delattr(test_module, "weight")
    with expect_raise(ValueError):
        torch_op.get_in_out_features(test_module)
