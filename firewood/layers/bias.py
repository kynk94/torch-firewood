import math
from typing import Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter

from firewood.common.types import FLOAT, INT
from firewood.layers import initializers


class Bias(nn.Module):
    def __init__(
        self,
        bias: Optional[Union[Tensor, INT, FLOAT]] = None,
        bias_gain: float = 1.0,
        size: Optional[INT] = None,
        initializer: str = "zeros",
    ) -> None:
        super().__init__()
        self.initializer = initializer
        if isinstance(bias, Tensor):
            self.bias = Parameter(bias)
        elif bias is not None:
            self.bias = Parameter(torch.tensor(bias, dtype=torch.float32))
        elif size is not None:
            if isinstance(size, Sequence):
                size = cast(Tuple[int, ...], size)
            self.bias = Parameter(torch.zeros(size, dtype=torch.float32))
            self.reset_parameters()
        else:
            self.register_parameter("bias", None)
        self.bias_gain = bias_gain

    def reset_parameters(self) -> None:
        if self.bias is None:
            return
        if self.initializer == "uniform":  # default torch init
            bound = 1 / math.sqrt(self.bias.shape[0])  # use fan_out as fan_in
            init.uniform_(self.bias, -bound, bound)
        else:
            initializers.get(self.initializer)(self.bias)

    def register_bias(
        self,
        bias: Tensor,
        bias_gain: Optional[float] = None,
    ) -> None:
        delattr(self, "bias")
        self.bias = Parameter(bias)
        if bias_gain is not None:
            self.bias_gain = bias_gain

    def forward(self, input: Tensor) -> Tensor:
        bias: Tensor = self.bias
        if bias is None:
            return input
        if self.bias_gain != 1.0:
            bias = bias * self.bias_gain
        bias = bias.view([-1 if i == 1 else 1 for i in range(input.ndim)])
        return input + bias.to(input.dtype)
