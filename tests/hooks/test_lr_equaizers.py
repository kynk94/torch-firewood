import torch
import torch.nn as nn
import torch.nn.functional as F

from firewood.hooks.lr_equalizers import lr_equalizer, remove_lr_equalizer
from firewood.layers.conv_blocks import Conv2dBlock


def test_assign_and_remove():
    kwargs = dict(bias=True, activation="lrelu")
    model = nn.Sequential(
        Conv2dBlock(3, 3, 3, 2, 1, **kwargs),
        Conv2dBlock(3, 3, 3, 2, 1, **kwargs),
    )
    lr_equalizer(model)

    for block in model:
        assert hasattr(block.layers["weighting"], "weight_param")
        assert hasattr(block.layers["activation"], "bias_param")

    remove_lr_equalizer(model)

    for block in model:
        if hasattr(block.layers["weighting"], "weight_param"):
            raise ValueError("weight_param found.")
        assert not hasattr(block.layers["activation"], "bias_param")


def test_with_nn():
    lr = 1e-2
    x = torch.randn(1, 3, 64, 64)
    x_nn = x.detach().clone()

    kwargs = dict(bias=False, normalization="bn", activation="lrelu")
    model = nn.Sequential(
        Conv2dBlock(3, 3, 3, 2, 1, **kwargs),
        Conv2dBlock(3, 3, 3, 2, 1, **kwargs),
    )
    lr_equalizer(model)

    class NNBlock(nn.Module):
        def __init__(self, weight, weight_gain):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, 2, 1, bias=False)
            self.conv.weight.data = weight.data
            self.normalization = nn.BatchNorm2d(3)
            self.activation = nn.LeakyReLU(0.2, inplace=True)
            self.weight_gain = weight_gain

        def forward(self, input):
            output = self.conv(input)
            output = output * self.weight_gain
            output = self.normalization(output)
            if self.activation is not None:
                output = self.activation(output)
            return output

    conv_1 = model[0].layers["weighting"]
    conv_2 = model[1].layers["weighting"]

    nn_model = nn.Sequential(
        NNBlock(conv_1.weight_param, conv_1._weight_coeff),
        NNBlock(conv_2.weight_param, conv_2._weight_coeff),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    nn_optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    optimizer.zero_grad()
    nn_optimizer.zero_grad()

    output = model(x)
    nn_output = nn_model(x_nn)
    loss = output.square().sum()
    nn_loss = nn_output.square().sum()

    loss.backward()
    nn_loss.backward()
    optimizer.step()
    nn_optimizer.step()

    assert torch.allclose(
        output, nn_output, rtol=1e-4, atol=1e-5
    ), f"Foward result mismatch. {F.l1_loss(output, nn_output)}"

    nn_weight_1 = nn_model[0].conv.weight
    assert torch.allclose(
        conv_1.weight_param, nn_weight_1
    ), f"Backward Weight mismatch. {F.l1_loss(conv_1.weight_param, nn_weight_1)}"
