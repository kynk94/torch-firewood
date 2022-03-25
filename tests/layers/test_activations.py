import pytest
import torch.nn as nn

from firewood.layers import activations
from firewood.layers.biased_activations import BiasedActivation
from firewood.layers.clamp import Clamp

AVAILABLE_ACTIVATIONS = {
    "linear": type(None),
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "threshold": nn.Threshold,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "selu": nn.SELU,
    "prelu": nn.PReLU,
}


@pytest.mark.xfail(raises=NotImplementedError)
def test_get_invalid_activation():
    activations.get("invalid")


def test_get_activation():
    for name, __class in AVAILABLE_ACTIVATIONS.items():
        kwargs = dict()
        if name == "threshold":
            kwargs.update(threshold=0.5, value=0.5)
        activation = activations.get(name, gain=1, **kwargs)
        assert isinstance(activation, __class)

    clamp = activations.get("linear", clamp=0.5)
    assert isinstance(clamp, Clamp)

    for activation in {"leaky_relu", "silu"}:
        biased_activation = activations.get(activation)
        assert isinstance(biased_activation, BiasedActivation)

    no_gain_activations = (
        set(AVAILABLE_ACTIVATIONS)
        - set(activations.SUPPORT_NN_INIT_GAIN)
        - set(activations.SUPPORT_CUSTOM_GAIN)
    )
    for name in no_gain_activations:
        kwargs = dict()
        if name == "threshold":
            kwargs.update(threshold=0.5, value=0.5)
        activation = activations.get(name, **kwargs)
        assert isinstance(activation, AVAILABLE_ACTIVATIONS[name])
