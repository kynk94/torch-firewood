import torch
import torch.nn.functional as F

from firewood.layers.bias import Bias


def test_bias() -> None:
    input = torch.randn(2, 3, 5, 5)
    bias = torch.randn(3)
    layer = Bias()
    layer.register_bias(bias)

    output = input + bias.view(1, -1, 1, 1)
    layer_output = layer(input)
    assert torch.allclose(
        output, layer_output
    ), f"tensor bias output mismatch. l1: {F.l1_loss(output, layer_output)}"

    for i, initializer in enumerate(("zeros", "ones", "uniform", "normal")):
        size = 3 if i % 2 == 0 else (3,)
        layer = Bias(size=size, initializer=initializer)
        layer.bias.data = bias
        layer_output = layer(input)
        assert torch.allclose(
            output, layer_output
        ), f"size bias output mismatch. l1: {F.l1_loss(output, layer_output)}"
