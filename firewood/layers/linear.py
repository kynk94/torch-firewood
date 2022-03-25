import math
from typing import Any, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init

from firewood.layers import initializers
from firewood.layers.block import Block


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        # parameter reset arguments
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def reset_parameters(self) -> None:
        # weight initialization
        # add other init methods here if needed
        if self.weight_initializer == "kaiming_uniform":  # default torch init
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            initializers.get(self.weight_initializer)(self.weight)

        if self.bias is None:
            return

        # bias initialization
        # add other init methods here if needed
        if self.bias_initializer == "uniform":  # default torch init
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            initializers.get(self.bias_initializer)(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return super().extra_repr() + ", ".join(
            [
                "",
                f"weight_initializer={self.weight_initializer}",
                f"bias_initializer={self.bias_initializer}",
            ]
        )


class LinearBlock(Block):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        op_order: str = "WNA",
        normalization: Optional[str] = None,
        normalization_args: Optional[Dict[str, Any]] = dict(),
        activation: str = None,
        activation_args: Optional[Dict[str, Any]] = dict(),
        lr_equalization: Optional[bool] = None,
        lr_equalization_args: Optional[Dict[str, Any]] = None,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        weight_layer = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )
        super().__init__(
            weight_layer=weight_layer,
            normalization=normalization,
            normalization_args=normalization_args,
            activation=activation,
            activation_args=activation_args,
            op_order=op_order,
            lr_equalization=lr_equalization,
            lr_equalization_args=lr_equalization_args,
        )
