from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from firewood.layers import normalizations
from firewood.layers.linear import Linear


class DeNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        normalization: str = "bn",
        eps: float = 1e-8,
        **normalization_kwargs: Any,
    ) -> None:
        super().__init__()
        if normalization is None:
            raise ValueError("normalization must be specified.")
        self.eps = eps
        self.normalization = normalizations.get(
            normalization=normalization,
            num_features=num_features,
            eps=self.eps,
            **normalization_kwargs,
        )

    def forward(
        self, input: Tensor, scale: Tensor, offset: Tensor, alpha: float = 1.0
    ) -> Tensor:
        output = self.normalization(input)
        output = output * scale + offset
        # TODO: support jit condition
        if alpha == 1.0:
            return output
        return alpha * output + (1.0 - alpha) * input


class AdaptiveNorm(DeNorm):
    """
    AdaIN: https://arxiv.org/abs/1703.06868

    If unbiased=True, use bessel's correction.
    Official implementation of AdaIN use bessel's correction.
    https://github.com/xunhuang1995/AdaIN-style
    """

    use_extra_inputs = True
    _unbiased = True

    def __init__(
        self,
        num_features: int,
        modulation_features: Optional[int] = None,
        normalization: str = "in",
        normalization_kwargs: Optional[dict] = None,
        unbiased: bool = True,
        use_projection: bool = False,
        use_separate_projection: bool = False,
        modulation_features_shape: Optional[Tuple[int, ...]] = None,
        eps: float = 1e-8,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        normalization_kwargs = normalization_kwargs or dict()
        normalization_kwargs.update(unbiased=unbiased)
        super().__init__(
            num_features=num_features,
            normalization=normalization,
            eps=eps,
            **normalization_kwargs,
        )
        self.unbiased = unbiased
        self.num_features = num_features
        self.modulation_features = modulation_features or num_features
        self.use_projection = use_projection or use_separate_projection
        self.use_separate_projection = use_separate_projection
        self.modulation_features_shape = modulation_features_shape

        if not self.use_projection:
            return

        if self.modulation_features_shape is None:
            self.modulation_features_shape = (self.modulation_features,)
        in_features = np.prod(self.modulation_features_shape)
        linear_kwargs: Any = dict(
            in_features=in_features,
            bias=True,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )
        if self.use_separate_projection:
            linear_kwargs["out_features"] = self.num_features
            self.gamma_linear = Linear(**linear_kwargs)
            self.beta_linear = Linear(**linear_kwargs)
        else:
            linear_kwargs["out_features"] = self.num_features * 2
            self.linear = Linear(**linear_kwargs)

    @property
    def unbiased(self) -> bool:
        return self._unbiased

    @unbiased.setter
    def unbiased(self, value: bool) -> None:
        self._unbiased = value
        if hasattr(self.normalization, "unbiased"):
            setattr(self.normalization, "unbiased", value)

    def forward(  # type: ignore
        self,
        input: Tensor,
        modulation_input: Tensor,
        alpha: float = 1.0,
    ) -> Tensor:
        if self.use_projection:
            modulation_input = modulation_input.view(
                modulation_input.size(0), -1
            )
            if self.use_separate_projection:
                gamma: Tensor = self.gamma_linear(modulation_input)
                beta: Tensor = self.beta_linear(modulation_input)
            else:
                adain_params: Tensor = self.linear(modulation_input)
                gamma, beta = adain_params.chunk(2, dim=1)
            for _ in range(input.ndim - gamma.ndim):
                gamma = gamma.unsqueeze(-1)
                beta = beta.unsqueeze(-1)
            gamma = gamma + 1.0
        else:
            assert input.ndim == modulation_input.ndim, (
                f"input.ndim={input.ndim} != "
                f"modulation_input.ndim={modulation_input.ndim}"
            )
            # TODO: support jit condition
            assert all(i != 1 for i in modulation_input.shape[2:])
            var, beta = torch.var_mean(
                modulation_input,
                dim=tuple(range(2, input.ndim)),
                unbiased=self.unbiased,
                keepdim=True,
            )
            gamma = (var + self.eps).sqrt()
        return super().forward(
            input=input, scale=gamma, offset=beta, alpha=alpha
        )

    def extra_repr(self) -> str:
        return super().extra_repr() + ", ".join(
            [
                f"unbiased={self.unbiased}",
                f"use_projection={self.use_projection}",
                f"use_separate_projection={self.use_separate_projection}",
            ]
        )


class SpatialAdaptiveNorm(DeNorm):
    """SPADE"""

    use_extra_inputs = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        kernel_size: Union[int, Tuple[int]] = 3,
        normalization: str = "sync_bn",
        normalization_kwargs: Optional[dict] = None,
        activation: str = "relu",
        eps: float = 1e-8,
        weight_initializer: str = "kaiming_uniform",
        bias_initializer: str = "zeros",
    ) -> None:
        super().__init__(
            num_features=in_channels,
            normalization=normalization,
            eps=eps,
            **normalization_kwargs or dict(),
        )

    def forward(  # type: ignore
        self,
        input: Tensor,
        modulation_input: Tensor,
        alpha: float = 1.0,
    ):
        # TODO: Implement SPADE.
        # return super().forward(input=input, scale=scale, offset=offset, alpha=alpha)
        raise NotImplementedError
