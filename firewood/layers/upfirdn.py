import functools
import math
import sys
from typing import Any, Callable, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch._C import Graph, Value

from firewood import utils
from firewood.common.constant import NULL_TENSOR
from firewood.common.types import DEVICE, INT, NUMBER
from firewood.functional.upfirdn import _parse_padding, upfirdnNd
from firewood.utils.extensions import CUDAExtension

FORCE_DEFAULT = False


def get_upfir_firdn_layers(
    rank: int,
    kernel: Optional[NUMBER] = None,
    up: Optional[INT] = 1,
    down: Optional[INT] = 1,
    padding: INT = 0,
    gain: float = 1.0,
    normalize_kernel: bool = True,
    flip_kernel: bool = False,
    upsample_mode: str = "zeros",
) -> Tuple[Optional["_UpFirDnNd"], Optional["_UpFirDnNd"]]:
    """
    Get upsample_fir and fir_downsample layers, not upfirdn, for the given rank.

    Because the basic order of layers is as follows, the returned layers are
    split into (upsample_fir, fir_downsample).
    Basic Order: upsample -> fir -> weighting -> fir -> downsample

    Args:
        upsample_mode: "zeros" or "nearest"
    """
    # `rank` == 0 means only linear layer, not 1x1 convolution, in Block.
    if rank == 0:
        return None, None
    upsample_fir_layer: Optional[_UpFirDnNd] = None
    fir_downsample_layer: Optional[_UpFirDnNd] = None
    up = utils.normalize_int_tuple(up or 1, rank)
    down = utils.normalize_int_tuple(down or 1, rank)
    module: _UpFirDnNd = getattr(sys.modules[__name__], f"UpFirDn{rank}d")
    if any(u > 1 for u in up):
        if upsample_mode.startswith("zero"):
            up_gain = gain * cast(float, np.prod(up))
        else:
            up_gain = gain
        upsample_fir_layer = module(
            kernel=kernel,
            up=up,
            down=1,
            padding=padding,
            gain=up_gain,
            normalize_kernel=normalize_kernel,
            flip_kernel=flip_kernel,
            ignore_same_padding=False,
        )
    if any(d > 1 for d in down):
        fir_downsample_layer = module(
            kernel=kernel,
            up=1,
            down=down,
            padding=padding,
            gain=gain,
            normalize_kernel=normalize_kernel,
            flip_kernel=flip_kernel,
            ignore_same_padding=False,
        )
    return upsample_fir_layer, fir_downsample_layer


class _UpFirDnNd(nn.Module):
    force_default = FORCE_DEFAULT
    use_resample = False
    kernel: Tensor

    __up: Tuple[int, ...] = (1,)
    __down: Tuple[int, ...] = (1,)

    def __init__(
        self,
        rank: int,
        kernel: Optional[NUMBER] = None,
        up: INT = 1,
        down: INT = 1,
        padding: INT = 0,
        gain: float = 1.0,
        normalize_kernel: bool = True,
        flip_kernel: bool = False,
        upsample_mode: str = "zeros",
        ignore_same_padding: bool = False,
        device: Optional[DEVICE] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")
        self.rank = rank
        self.up = utils.normalize_int_tuple(up, rank)
        self.down = utils.normalize_int_tuple(down, rank)
        self.gain = gain
        self.normalize_kernel = normalize_kernel
        self.flip_kernel = flip_kernel
        self.upsample_mode = upsample_mode
        self.ignore_same_padding = ignore_same_padding

        self.register_buffer(
            name="kernel",
            tensor=_setup_kernel(
                rank=self.rank,
                kernel=kernel,
                gain=1.0,
                normalize_kernel=self.normalize_kernel,
                flip_kernel=self.flip_kernel,
            ),
        )
        self.use_fir = self.kernel.numel() != 1 or self.kernel.item() != 1.0
        if not self.use_fir and not self.use_resample:
            raise ValueError(
                "Both FIR and resampling are disabled. Should remove this layer."
            )

        padding = _parse_padding(rank=rank, padding=padding)
        if not self.ignore_same_padding:
            same_padding = _calc_padding(
                rank=rank,
                kernel_size=cast(INT, self.kernel.shape),
                up=self.up,
                down=self.down,
            )
            padding = tuple(p + s_p for p, s_p in zip(padding, same_padding))
        self.padding = padding
        self.default_operation = functools.partial(
            upfirdnNd,
            gain=self.gain,
            flip_kernel=self.flip_kernel,
            up=self.up,
            down=self.down,
            padding=self.padding,
            upsample_mode=self.upsample_mode,
        )

    def _apply(self, fn: Callable[..., Any]) -> "_UpFirDnNd":
        if "t" in fn.__code__.co_varnames:
            with torch.no_grad():
                device = getattr(fn(NULL_TENSOR), "device", "cpu")
            self.device = torch.device(device)
        return super()._apply(fn)

    @property
    def up(self) -> Tuple[int, ...]:
        return self.__up

    @up.setter
    def up(self, value: INT) -> None:
        self.__up = utils.normalize_int_tuple(value, self.rank)
        self.use_resample = any(factor > 1 for factor in self.up + self.down)

    @property
    def down(self) -> Tuple[int, ...]:
        return self.__down

    @down.setter
    def down(self, value: INT) -> None:
        self.__down = utils.normalize_int_tuple(value, self.rank)
        self.use_resample = any(factor > 1 for factor in self.up + self.down)

    @property
    def operation(self) -> Callable[..., Tensor]:
        if (
            self.device.type != "cuda"
            or not self.use_resample
            or self.force_default
        ):
            return self.default_operation
        try:
            return load_cuda_upfirdn2d(
                up=self.up,
                down=self.down,
                padding=self.padding,
                flip_kernel=self.flip_kernel,
                gain=self.gain,
            )
        except RuntimeError:
            self.force_default = True
            return self.default_operation

    def forward(self, input: Tensor) -> Tensor:
        return self.operation(input, self.kernel)

    def extra_repr(self) -> str:
        s = []
        if any(f != 1 for f in self.up):
            s.append(f"up={self.up}")
        if any(f != 1 for f in self.down):
            s.append(f"down={self.down}")
        s.extend(
            [
                f"gain={self.gain}",
                f"normalize_kernel={self.normalize_kernel}",
                f"flip_kernel={self.flip_kernel}",
            ]
        )
        return ", ".join(s)


def _setup_kernel(
    rank: int,
    kernel: Optional[NUMBER] = None,
    gain: float = 1.0,
    normalize_kernel: bool = True,
    flip_kernel: bool = False,
    separable: Optional[bool] = None,
) -> Tensor:
    if isinstance(kernel, Tensor):
        kernel = kernel.float()
    elif isinstance(kernel, np.ndarray):
        kernel = torch.as_tensor(kernel, dtype=torch.float32)
    else:
        kernel = torch.as_tensor(kernel or 1, dtype=torch.float32)

    if kernel.numel() == 0:
        raise ValueError("Kernel must have at least one element.")

    if kernel.ndim == 0:
        kernel.unsqueeze_(0)

    if separable is None:
        separable = kernel.ndim == 1 and kernel.numel() >= 8

    if kernel.ndim < rank and not separable:
        kernels = []
        ein_dims, ein_source, ein_target = "jkl", [], []
        for i in range(rank):
            shape = [1] * rank
            shape[i] = -1
            kernels.append(kernel.view(shape))
            base = ["i"] * rank
            base[i] = ein_dims[i]
            ein_source.append("".join(base))
            ein_target.append(ein_dims[i])
        einstr = ",".join(ein_source) + "->" + "".join(ein_target)
        kernel = torch.einsum(einstr, *kernels)

    if normalize_kernel:
        kernel = kernel / kernel.abs().sum()
    if flip_kernel:
        kernel = kernel.flip(list(range(kernel.ndim)))
    kernel *= gain ** (kernel.ndim / rank)
    return kernel


def _calc_padding(
    rank: int,
    kernel_size: INT,
    up: INT,
    down: INT,
) -> Tuple[int, ...]:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * rank
    elif len(kernel_size) == 1:
        kernel_size = (kernel_size[0],) * rank
    if len(kernel_size) != rank:
        raise ValueError(
            f"kernel_size must be either an integer or an iterable of length {rank}."
        )
    up = utils.normalize_int_tuple(up, rank)
    down = utils.normalize_int_tuple(down, rank)
    padding = []
    for k, u, d in zip(kernel_size, up, down):
        if u == d == 1:
            div, mod = divmod(k - 1, 2)
            padding.extend([div + mod, div])
            continue
        p_0, p_1 = 0, 0
        if u > 1:
            p_0 += (k + u - 1) // 2
            p_1 += (k - u) // 2
        if d > 1:
            p_0 += (k - d + 1) // 2
            p_1 += (k - d) // 2
        padding.extend([p_0, p_1])
    return tuple(padding)


def load_cuda_upfirdn2d(
    up: Tuple[int, ...],
    down: Tuple[int, ...],
    padding: Tuple[int, ...],
    flip_kernel: bool,
    gain: float = 1.0,
) -> Callable[..., Tensor]:
    name = "upfirdn2d"
    cache_key = (up, down, padding, flip_kernel, gain)
    cache = CUDAExtension.cache(name)
    if cache_key in cache:
        return cache[cache_key].apply

    cuda_extension = CUDAExtension.get(name)

    gain *= cast(float, np.prod(up))

    class UpFirDn2dCUDA(torch.autograd.Function):
        @staticmethod
        # type: ignore[override]
        def forward(ctx: Any, input: Tensor, kernel: Tensor) -> Tensor:
            output = input
            ctx.save_for_backward(kernel)
            ctx.input_shape = input.shape

            if kernel.ndim == 2:
                output = cuda_extension(
                    x=output,
                    f=kernel,
                    upx=up[0],
                    upy=up[1],
                    downx=down[0],
                    downy=down[1],
                    padx0=padding[0],
                    padx1=padding[1],
                    pady0=padding[2],
                    pady1=padding[3],
                    flip=flip_kernel,
                    gain=gain,
                )
            else:
                output = cuda_extension(
                    x=output,
                    f=kernel.unsqueeze(0),
                    upx=up[0],
                    upy=1,
                    downx=down[0],
                    downy=1,
                    padx0=padding[0],
                    padx1=padding[1],
                    pady0=0,
                    pady1=0,
                    flip=flip_kernel,
                    gain=math.sqrt(gain),
                )
                output = cuda_extension(
                    x=output,
                    f=kernel.unsqueeze(1),
                    upx=1,
                    upy=up[1],
                    downx=1,
                    downy=down[1],
                    padx0=0,
                    padx1=0,
                    pady0=padding[2],
                    pady1=padding[3],
                    flip=flip_kernel,
                    gain=math.sqrt(gain),
                )
            return output

        @staticmethod
        def symbolic(g: Graph, input: Value, kernel: Value) -> None:
            raise NotImplementedError("Use default version, not CUDA.")

        @staticmethod
        # type: ignore[override]
        def backward(
            ctx: Any, grad_output: Tensor
        ) -> Tuple[Optional[Tensor], None]:
            (kernel,) = ctx.saved_tensors
            *_, input_H, input_W = ctx.input_shape
            *_, output_H, output_W = grad_output.shape
            kernel_H, *_, kernel_W = kernel.shape

            grad_padding = (
                kernel_W - padding[0] - 1,
                (input_W - 1) * up[0] - output_W * down[0] + padding[0] + 1,
                kernel_H - padding[2] - 1,
                (input_H - 1) * up[1] - output_H * down[1] + padding[2] + 1,
            )
            grad_input = None
            grad_kernel = None

            if ctx.needs_input_grad[0]:
                grad_input = load_cuda_upfirdn2d(
                    up=down,
                    down=up,
                    padding=grad_padding,
                    flip_kernel=not flip_kernel,
                )(grad_output, kernel)
            return grad_input, grad_kernel

    cache[cache_key] = UpFirDn2dCUDA
    return UpFirDn2dCUDA.apply


class UpFirDn1d(_UpFirDnNd):
    def __init__(
        self,
        kernel: Optional[NUMBER] = None,
        up: INT = 1,
        down: INT = 1,
        padding: INT = 0,
        gain: float = 1.0,
        normalize_kernel: bool = True,
        flip_kernel: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rank=1,
            kernel=kernel,
            up=up,
            down=down,
            padding=padding,
            gain=gain,
            normalize_kernel=normalize_kernel,
            flip_kernel=flip_kernel,
            **kwargs,
        )


class UpFirDn2d(_UpFirDnNd):
    def __init__(
        self,
        kernel: Optional[NUMBER] = None,
        up: INT = 1,
        down: INT = 1,
        padding: INT = 0,
        gain: float = 1.0,
        normalize_kernel: bool = True,
        flip_kernel: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rank=2,
            kernel=kernel,
            up=up,
            down=down,
            padding=padding,
            gain=gain,
            normalize_kernel=normalize_kernel,
            flip_kernel=flip_kernel,
            **kwargs,
        )


class UpFirDn3d(_UpFirDnNd):
    def __init__(
        self,
        kernel: Optional[NUMBER] = None,
        up: INT = 1,
        down: INT = 1,
        padding: INT = 0,
        gain: float = 1.0,
        normalize_kernel: bool = True,
        flip_kernel: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            rank=3,
            kernel=kernel,
            up=up,
            down=down,
            padding=padding,
            gain=gain,
            normalize_kernel=normalize_kernel,
            flip_kernel=flip_kernel,
            **kwargs,
        )
