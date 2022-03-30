import functools
import math
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn.parameter import Parameter

from firewood import utils
from firewood.common.backend import runtime_build
from firewood.common.constant import NULL_TENSOR
from firewood.common.types import DEVICE, FLOAT, INT
from firewood.utils.image import nearest_downsample, upsample

_CUDA_OPERATION_CACHE: Dict[
    Tuple[
        Tuple[int, ...],  # up
        Tuple[int, ...],  # down
        Tuple[int, ...],  # padding
        bool,  # flip_kernel
        float,  # gain
    ],
    Type[torch.autograd.Function],
] = dict()
_PRE_BUILD_CUDA_UPFIRDN_EXTENSION = None
_RUNTIME_BUILD_CUDA_UPFIRDN_EXTENSION = None


def _load_runtime_cuda_extension() -> Optional[Callable[..., Tensor]]:
    global _CUDA_OPERATION_CACHE
    global _PRE_BUILD_CUDA_UPFIRDN_EXTENSION
    global _RUNTIME_BUILD_CUDA_UPFIRDN_EXTENSION
    if _RUNTIME_BUILD_CUDA_UPFIRDN_EXTENSION is None:
        _PRE_BUILD_CUDA_UPFIRDN_EXTENSION = None
        _CUDA_OPERATION_CACHE.clear()
        try:
            _C = utils.extensions.load_cuda_extension()
            _RUNTIME_BUILD_CUDA_UPFIRDN_EXTENSION = _C.upfirdn2d
        except RuntimeError:
            warnings.warn("CUDA extension not found. Load default operation.")
    return _RUNTIME_BUILD_CUDA_UPFIRDN_EXTENSION


def _load_cuda_extension() -> Optional[Callable[..., Tensor]]:
    global _CUDA_OPERATION_CACHE
    global _PRE_BUILD_CUDA_UPFIRDN_EXTENSION
    global _RUNTIME_BUILD_CUDA_UPFIRDN_EXTENSION

    if _RUNTIME_BUILD_CUDA_UPFIRDN_EXTENSION is not None:
        return _RUNTIME_BUILD_CUDA_UPFIRDN_EXTENSION

    if runtime_build():
        return _load_runtime_cuda_extension()
    if _PRE_BUILD_CUDA_UPFIRDN_EXTENSION is None:
        _RUNTIME_BUILD_CUDA_UPFIRDN_EXTENSION = None
        _CUDA_OPERATION_CACHE.clear()
        try:
            from firewood._C import upfirdn2d

            _PRE_BUILD_CUDA_UPFIRDN_EXTENSION = upfirdn2d
        except (RuntimeError, ModuleNotFoundError):
            warnings.warn(
                "Pre-build CUDA extension not found. Load default operation."
            )
            return _load_runtime_cuda_extension()
    return _PRE_BUILD_CUDA_UPFIRDN_EXTENSION


def get_upfirdn_layer(
    rank: int,
    kernel: Optional[Union[INT, FLOAT, np.ndarray, Tensor]] = None,
    up: INT = 1,
    down: INT = 1,
    padding: INT = 0,
    gain: float = 1.0,
    normalize_kernel: bool = True,
    flip_kernel: bool = False,
    device: Optional[DEVICE] = None,
) -> Tuple[Optional["_UpFirDnNd"], Optional["_UpFirDnNd"]]:
    upfir: Optional[_UpFirDnNd] = None
    firdown: Optional[_UpFirDnNd] = None
    up = utils.normalize_int_tuple(up, rank)
    down = utils.normalize_int_tuple(down, rank)
    if any(u > 1 for u in up):
        upfir = _UpFirDnNd(
            rank=rank,
            kernel=kernel,
            up=up,
            down=1,
            padding=padding,
            gain=gain,
            normalize_kernel=normalize_kernel,
            flip_kernel=flip_kernel,
            ignore_same_padding=False,
            device=device,
        )
        padding = 0
    if any(d > 1 for d in down) or (upfir is None and kernel is not None):
        firdown = _UpFirDnNd(
            rank=rank,
            kernel=kernel,
            up=1,
            down=down,
            padding=padding,
            gain=gain,
            normalize_kernel=normalize_kernel,
            flip_kernel=flip_kernel,
            ignore_same_padding=False,
            device=device,
        )
    return upfir, firdown


class _UpFirDnNd(nn.Module):
    operation: Callable[..., Tensor]

    def __init__(
        self,
        rank: int,
        kernel: Optional[Union[INT, FLOAT, np.ndarray, Tensor]] = None,
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

        self.kernel = Parameter(
            _setup_kernel(
                rank=self.rank,
                kernel=kernel,
                gain=1.0,
                normalize_kernel=self.normalize_kernel,
                flip_kernel=self.flip_kernel,
            ),
            requires_grad=False,
        )

        padding = _parse_padding(rank=rank, padding=padding)
        if not self.ignore_same_padding:
            same_padding = _calc_padding(
                rank=rank,
                kernel_size=self.kernel.shape,
                up=self.up,
                down=self.down,
            )
            padding = tuple(p + s for p, s in zip(padding, same_padding))
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
        self.to(device=self.device)

    def forward(self, input: Tensor) -> Tensor:
        return self.operation(input, self.kernel)

    def _cpu(self) -> None:
        self.device = torch.device("cpu")
        setattr(self, "operation", self.default_operation)

    def _cuda(self) -> None:
        self.device = torch.device(torch.cuda.current_device())
        if self.rank != 2:
            return
        try:
            cuda_operation = load_cuda_upfirdn2d(
                up=self.up,
                down=self.down,
                padding=self.padding,
                flip_kernel=self.flip_kernel,
                gain=self.gain,
            )
            setattr(self, "operation", cuda_operation)
        except RuntimeError as e:
            print(e)

    def _apply(self, fn: Callable[..., Any]) -> Any:
        if "t" in fn.__code__.co_varnames:
            with torch.no_grad():
                device = getattr(fn(NULL_TENSOR), "device", "cpu")
            if utils.is_cuda(device):
                self._cuda()
            else:
                self._cpu()
        return super()._apply(fn)

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
    kernel: Union[INT, FLOAT, np.ndarray, Tensor] = None,
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
        kernel = cast(Tensor, torch.einsum(einstr, *kernels))

    if normalize_kernel:
        kernel = kernel / kernel.abs().sum()
    if flip_kernel:
        kernel = kernel.flip(list(range(kernel.ndim)))
    kernel *= gain ** (kernel.ndim / rank)
    return kernel


def _parse_padding(rank: int, padding: INT) -> Tuple[int, ...]:
    if isinstance(padding, int):
        padding = (padding,) * rank * 2
    elif len(padding) == 1:
        padding = (padding[0],) * rank * 2
    elif len(padding) == rank:
        padding = cast(Tuple[int, ...], _reverse_repeat_tuple(padding, 2))
    elif len(padding) == rank * 2:
        padding = tuple(padding)
    else:
        raise ValueError(
            f"Padding must be either integer or iterable of length {rank} or {rank * 2}."
        )
    return padding


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


def firNd(
    input: Tensor, kernel: Tensor, gain: float = 1.0, flip_kernel: bool = False
) -> Tensor:
    rank = input.ndim - 2
    if rank == 1:
        conv = F.conv1d
    elif rank == 2:
        conv = F.conv2d
    elif rank == 3:
        conv = F.conv3d
    else:
        raise ValueError(f"Rank {rank} is not supported.")

    if not flip_kernel:
        kernel = kernel.flip(list(range(kernel.ndim)))

    C = input.size(1)
    if gain != 1.0:
        kernel = kernel * gain ** (kernel.ndim / rank)
    kernel = kernel.view(1, 1, *kernel.shape).repeat(C, 1, *(1,) * kernel.ndim)

    if kernel.ndim == input.ndim:
        return conv(input, kernel, groups=C)
    output = input
    for i in range(2, rank + 2):
        output = conv(output, kernel.unsqueeze(i), groups=C)
    return output


def upfirdnNd(
    input: Tensor,
    kernel: Tensor,
    gain: float = 1.0,
    flip_kernel: bool = False,
    up: INT = 1,
    down: INT = 1,
    padding: INT = 0,
    upsample_mode: str = "zeros",
) -> Tensor:
    if isinstance(up, int):
        up = (up,) * (input.ndim - 2)
    if isinstance(down, int):
        down = (down,) * (input.ndim - 2)
    if isinstance(padding, int):
        padding = (padding,) * (input.ndim - 2)

    # upsample
    if any(u > 1 for u in up) and all(u >= 1 for u in up):
        input = upsample(input, up, mode=upsample_mode)
    if upsample_mode.startswith("zero"):
        gain = gain * math.prod(up)

    # pad
    if any(p != 0 for p in padding):
        input = F.pad(input, padding, mode="constant", value=0)

    # fir
    output = firNd(input, kernel, gain=gain, flip_kernel=flip_kernel)

    # downsample
    if any(d > 1 for d in down) and all(d >= 1 for d in down):
        output = nearest_downsample(output, down)
    return output


def load_cuda_upfirdn2d(
    up: Tuple[int, ...],
    down: Tuple[int, ...],
    padding: Tuple[int, ...],
    flip_kernel: bool,
    gain: float = 1.0,
) -> Callable[..., Tensor]:
    cache_key = (up, down, padding, flip_kernel, gain)
    if cache_key in _CUDA_OPERATION_CACHE:
        return _CUDA_OPERATION_CACHE[cache_key].apply

    cuda_extension = _load_cuda_extension()
    if cuda_extension is None:
        raise RuntimeError(
            "CUDA extension could not be loaded. "
            "Make sure that the CUDA extension is installed and that "
            "the CUDA_HOME environment variable is set."
        )

    gain *= math.prod(up)

    class UpFirDn2dCUDA(torch.autograd.Function):
        @staticmethod
        # type: ignore[override]
        def forward(ctx: Any, input: Tensor, kernel: Tensor) -> Tensor:
            output = input
            kernel = kernel.to(input.dtype)
            ctx.save_for_backward(kernel)
            ctx.input_shape = input.shape

            if kernel.ndim == 2:
                output = cuda_extension(  # type: ignore
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
                output = cuda_extension(  # type: ignore
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
                output = cuda_extension(  # type: ignore
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

    _CUDA_OPERATION_CACHE[cache_key] = UpFirDn2dCUDA
    return UpFirDn2dCUDA.apply


class UpFirDn1d(_UpFirDnNd):
    def __init__(
        self,
        kernel: Optional[Union[INT, FLOAT, np.ndarray, Tensor]] = None,
        up: INT = 1,
        down: INT = 1,
        padding: INT = 0,
        gain: float = 1.0,
        normalize_kernel: bool = True,
        flip_kernel: bool = False,
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
        )


class UpFirDn2d(_UpFirDnNd):
    def __init__(
        self,
        kernel: Optional[Union[INT, FLOAT, np.ndarray, Tensor]] = None,
        up: INT = 1,
        down: INT = 1,
        padding: INT = 0,
        gain: float = 1.0,
        normalize_kernel: bool = True,
        flip_kernel: bool = False,
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
        )


class UpFirDn3d(_UpFirDnNd):
    def __init__(
        self,
        kernel: Optional[Union[INT, FLOAT, np.ndarray, Tensor]] = None,
        up: INT = 1,
        down: INT = 1,
        padding: INT = 0,
        gain: float = 1.0,
        normalize_kernel: bool = True,
        flip_kernel: bool = False,
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
        )
