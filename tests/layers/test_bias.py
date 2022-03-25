import torch
import torch.nn.functional as F

from firewood.layers.bias import Bias


def test_bias() -> None:
    input = torch.randn(2, 3, 5, 5)
    bias = torch.randn(3)
    layer = Bias(bias)

    output = input + bias.view(1, -1, 1, 1)
    layer_output = layer(input)
    assert torch.allclose(
        output, layer_output
    ), f"tensor bias output mismatch. l1: {F.l1_loss(output, layer_output)}"

    layer = Bias(bias.numpy())
    layer_output = layer(input)
    assert torch.allclose(
        output, layer_output
    ), f"numpy bias output mismatch. l1: {F.l1_loss(output, layer_output)}"

    for i, initializer in enumerate(("zeros", "ones", "uniform", "normal")):
        size = 3 if i % 2 == 0 else (3,)
        layer = Bias(size=size, initializer=initializer)
        layer.bias.data = bias
        layer_output = layer(input)
        assert torch.allclose(
            output, layer_output
        ), f"size bias output mismatch. l1: {F.l1_loss(output, layer_output)}"

    layer = Bias()
    layer.reset_parameters()
    layer_output = layer(input)
    assert torch.allclose(
        input, layer_output
    ), f"default bias output mismatch. l1: {F.l1_loss(input, layer_output)}"

    layer.register_bias(bias, bias_gain=1.0)
    layer_output = layer(input)
    assert torch.allclose(
        output, layer_output
    ), f"register bias output mismatch. l1: {F.l1_loss(output, layer_output)}"

    gain = 0.5
    gain_multiplied_output = input + gain * bias.view(1, -1, 1, 1)
    layer.register_bias(bias, bias_gain=gain)
    layer_output = layer(input)
    assert torch.allclose(
        gain_multiplied_output, layer_output
    ), f"gain multiplied bias output mismatch. l1: {F.l1_loss(gain_multiplied_output, layer_output)}"
