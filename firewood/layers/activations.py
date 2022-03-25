import math
from typing import Any, Optional, cast

import torch.nn as nn

from firewood import utils
from firewood.layers.biased_activations import BiasedActivation
from firewood.layers.clamp import Clamp

SUPPORT_NN_INIT_GAIN = {
    "linear",
    "relu",
    "leaky_relu",
    "tanh",
    "sigmoid",
    "softmax",
    "selu",
}
SUPPORT_CUSTOM_GAIN = {"silu": math.sqrt(2.0)}


def get(
    activation: Optional[str],
    alpha: Optional[float] = None,
    gain: Optional[float] = None,
    clamp: Optional[float] = None,
    inplace: bool = True,
    bias_gain: float = 1.0,
    **kwargs: Any,
) -> Optional[nn.Module]:
    activation = utils.normalize_activation_name(activation)
    if activation in {"leaky_relu", "elu"} and alpha is None:
        alpha = 0.2 if activation == "leaky_relu" else 1.0
    if gain is None:
        if activation in SUPPORT_NN_INIT_GAIN:
            gain = nn.init.calculate_gain(activation, alpha)
        elif activation in SUPPORT_CUSTOM_GAIN:
            gain = SUPPORT_CUSTOM_GAIN[activation]
        else:
            gain = 1.0
    clamp = -1.0 if clamp is None else clamp
    if gain != 1.0:
        # Biased activation is runtime weight scaling operation.
        # It is faster when multiplying by gain.
        # Without gain multiplication, basic operations are faster.
        return BiasedActivation(
            activation=activation,
            alpha=alpha,
            gain=gain,
            bias_gain=bias_gain,
            clamp=clamp,
        )

    # Belows are 'gain = 1'.
    # They operate on tensors without runtime weight scaling.
    # Input tensors are multiplied by weight which multiplied by weight gain.
    if activation == "linear":
        if clamp < 0:
            return None
        return Clamp(min=-clamp, max=clamp)
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    if activation == "leaky_relu":
        alpha = cast(float, alpha)
        return nn.LeakyReLU(negative_slope=alpha, inplace=inplace)
    if activation == "tanh":
        return nn.Tanh()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "softmax":
        return nn.Softmax(dim=-1)
    if activation == "softplus":
        return nn.Softplus()
    if activation == "elu":
        alpha = cast(float, alpha)
        return nn.ELU(alpha=alpha, inplace=inplace)
    if activation == "silu":
        return nn.SiLU(inplace=inplace)
    if activation == "selu":
        return nn.SELU(inplace=inplace)
    if activation == "prelu":
        return nn.PReLU(**kwargs)
    if activation == "threshold":
        return nn.Threshold(inplace=inplace, **kwargs)
    raise NotImplementedError(
        f"Received activation is not implemented. Received: {activation}"
    )
