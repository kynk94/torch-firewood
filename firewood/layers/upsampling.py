from typing import Optional, cast

import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _ratio_any_t, _size_any_t

from firewood.common.types import INT
from firewood.functional.resample import zero_insertion_upsample


class Upsample(nn.Upsample):
    """
    Upsample the input, inheriting from `nn.Upsample`.
    This supports zero insertion upsampling additionally, which is not supported
    by `nn.Upsample`.

    Args:
        ... (see `nn.Upsample`)
        size: ...
        scale_factor: ...
        mode: { "nearest", "zero", "linear", "bilinear", "bicubic", "trilinear" }
            Mode "zero" is zero insertion upsampling. If mode is "zero", `size`
            argument does not supported and `scale_factor` must be integer.
            Default: "nearest".
        align_corners: ...
        recompute_scale_factor: ...
        gain: multiplier for the output. Default: 1.0.
    """

    def __init__(
        self,
        size: Optional[_size_any_t] = None,
        scale_factor: Optional[_ratio_any_t] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        gain: float = 1.0,
    ) -> None:
        if mode.startswith("zero"):
            mode = "zero"
        super().__init__(
            size, scale_factor, mode, align_corners, recompute_scale_factor
        )
        self.gain = gain

        if self.mode == "zero":
            if size is not None:
                raise ValueError(
                    "`size` argument is not supported for `mode` 'zero'."
                )
            if (
                isinstance(self.scale_factor, float)
                and self.scale_factor.is_integer()
            ):
                self.scale_factor = int(self.scale_factor)
            elif isinstance(self.scale_factor, tuple) and all(
                s.is_integer() for s in self.scale_factor
            ):
                self.scale_factor = tuple(int(s) for s in self.scale_factor)
            else:
                raise ValueError(
                    "`scale_factor` must be integer for `mode` 'zero'."
                )

    def forward(self, input: Tensor) -> Tensor:
        if self.mode == "zero":
            output = zero_insertion_upsample(
                input, cast(INT, self.scale_factor)
            )
        else:
            output = super().forward(input)
        if self.gain != 1.0:
            output = output * self.gain
        return output
