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
        size: Optional[INT] = None,
        bias_add_dim: int = 1,
        initializer: str = "zeros",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.bias_add_dim = bias_add_dim
        self.initializer = initializer
        self.dtype = dtype or torch.float32
        if size is not None:
            if isinstance(size, Sequence):
                size = cast(Tuple[int, ...], size)
            self.bias = Parameter(torch.zeros(size, dtype=self.dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

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
        bias: Union[Tensor, INT, FLOAT],
    ) -> None:
        if bias is None:
            raise ValueError("bias cannot be None")
        delattr(self, "bias")
        if isinstance(bias, Tensor):
            self.bias = Parameter(bias.detach().to(dtype=self.dtype))
        else:
            self.bias = Parameter(torch.tensor(bias, dtype=self.dtype))

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None:
            raise ValueError(
                "bias is not registered. Call `register_bias` first."
            )
        bias = self.bias.view(
            [-1 if i == self.bias_add_dim else 1 for i in range(input.ndim)]
        )
        return input + bias.to(dtype=input.dtype)

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"size={self.bias.shape[0] if self.bias is not None else None}",
                f"bias_add_dim={self.bias_add_dim}",
                f"initializer={self.initializer}",
                f"dtype={self.dtype}",
            ]
        )
