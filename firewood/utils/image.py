import os
from typing import Any, Optional, Sequence, Tuple, Union, cast, overload

import imageio
import numpy as np
import torch
import torchvision.transforms.functional_tensor as TFT
import torchvision.utils as TU
from numpy import ndarray
from PIL import Image
from torch import Tensor

from firewood.common.types import INT
from firewood.utils.common import (
    normalize_float_tuple,
    normalize_int_tuple,
    squared_number,
)


def alpha_smoothing(
    image: Union[ndarray, Image.Image],
    alpha: Optional[Union[ndarray, Image.Image]] = None,
    background_color: Union[ndarray, INT] = 0,
) -> ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cast(ndarray, image.astype(np.float32))
    if image.ndim == 2:
        original_dim = 2
        image = np.expand_dims(image, axis=-1)
    else:
        original_dim = 3

    H, W, C = image.shape
    if alpha is None:
        if C not in {2, 4}:
            raise ValueError(
                "If alpha is None, image should have 2 or 4 channels."
            )
        alpha = image[..., -1]
        image = image[..., :-1]
        C -= 1
    elif isinstance(alpha, Image.Image):
        if C not in {1, 3}:
            raise ValueError(
                "If alpha exists, image should have 1 or 3 channels."
            )
        if alpha.size != (W, H):
            alpha = alpha.resize((W, H))
        alpha = np.array(alpha, dtype=np.float32)
    elif alpha.shape[:2] != (H, W):
        raise ValueError(
            "If alpha exists, it should have the same shape as image."
        )
    alpha = normalize_image_array(cast(ndarray, alpha), 0, 1)
    if alpha.ndim == 2:
        alpha = np.expand_dims(alpha, -1)

    if not isinstance(background_color, ndarray):
        background_color = np.array(
            normalize_float_tuple(background_color, 3), dtype=np.float32
        )
    background_color = background_color.reshape(-1).astype(np.float32)
    if background_color.shape[-1] not in {1, 3}:
        raise ValueError(
            "background_color should be an iterable of length 1 or 3."
        )
    if C == 1 and background_color.shape[-1] == 3:
        background_color = np.matmul(
            background_color,
            np.array([0.299, 0.587, 0.114], dtype=np.float32),
        )
    background = np.ones_like(image) * background_color
    output = alpha * image + (1 - alpha) * background
    if original_dim == 2:
        return output[..., 0]
    return output


def alpha_smoothing_NCHW_tensor(
    image: Tensor,
    alpha: Optional[Tensor] = None,
    background_color: Union[Tensor, INT] = 0,
    antialias: bool = False,
) -> Tensor:
    device = image.device
    if image.dtype != torch.float32:
        image = image.float()

    C, H, W = image.shape[1:]
    if alpha is None:
        if C not in {2, 4}:
            raise ValueError(
                "If alpha is None, image should have 2 or 4 channels."
            )
        alpha = image[:, -1]
        image = image[:, :-1]
        C -= 1
    elif C not in {1, 3}:
        raise ValueError("If alpha exists, image should have 1 or 3 channels.")
    elif alpha.shape[2:] != (H, W):
        alpha = TFT.resize(
            alpha.float(),
            (H, W),
            interpolation="bilinear",
            antialias=antialias,  # antialias does not keep gradient
        )
    alpha = normalize_image_array(
        cast(Tensor, alpha).to(device=device, non_blocking=True), 0, 1
    )

    if not isinstance(background_color, Tensor):
        background_color = torch.tensor(
            normalize_float_tuple(background_color, 3), dtype=torch.float32
        )
    background_color = background_color.float().squeeze().flatten(-1)
    if background_color.ndim > 2:
        raise ValueError("background_color should be a 1D or 2D tensor.")
    if background_color.size(-1) not in {1, 3}:
        raise ValueError(
            "background_color should be an iterable of length 1 or 3."
        )
    if C == 1 and background_color.size(-1) == 3:
        background_color = torch.matmul(
            background_color,
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32),
        )
    background = background_color.view(-1, C, 1, 1).to(device=device)
    return torch.lerp(background, image, alpha)


def make_grid(
    tensor: Union[Tensor, ndarray],
    nrow: Optional[int] = None,
    padding: int = 0,
    normalize: bool = True,
    value_range: Tuple[int, ...] = (-1, 1),
    dataformats: str = "NCHW",
) -> ndarray:
    if isinstance(tensor, ndarray):
        tensor = torch.from_numpy(tensor)

    if nrow is None:
        batch = tensor.shape[0]
        nrow = int(batch**0.5)

    lower_dataformats = dataformats.strip().lower()
    if lower_dataformats in {"channels_last", "nhwc"}:
        tensor = tensor.permute(0, 3, 1, 2)
        transpose = True
    elif lower_dataformats in {"channels_first", "nchw"}:
        transpose = False
    else:
        raise ValueError(
            "dataformats should be one of "
            "{channels_first, channels_last, NCHW, NHWC}. "
            f"Recieved: {dataformats}"
        )

    image_grid: np.ndarray = (
        TU.make_grid(
            tensor=tensor,
            nrow=nrow,
            padding=padding,
            normalize=normalize,
            value_range=value_range,
            scale_each=False,
            pad_value=0,
        )
        .mul_(255)
        .add_(0.5)
        .clamp_(0, 255)
        .to(dtype=torch.uint8, device="cpu")
        .numpy()
    )

    if transpose:
        return image_grid.transpose(1, 2, 0)
    return image_grid


def image_auto_permute(
    tensor: Tensor, target_format: str = "NHWC", ignore_error: bool = False
) -> Tensor:
    target_format = target_format.lower()
    if target_format.endswith("hwc") or target_format.endswith("last"):
        target_format = "channels_last"
    elif target_format.endswith("chw") or target_format.endswith("first"):
        target_format = "channels_first"
    else:
        raise ValueError(f"Invalid target_format: {target_format}")
    sizes = []
    possible_channel_dims = []
    possible_alpha_channel_dims = []
    for dim in range(tensor.ndim - 3, tensor.ndim):
        size = tensor.size(dim)
        sizes.append(size)
        if size in {1, 3}:
            possible_channel_dims.append(dim)
        elif size == 4:
            possible_alpha_channel_dims.append(dim)
    if sizes[0] == sizes[1] == sizes[2] and ignore_error:
        return tensor
    if sizes[0] == sizes[1] and sizes[1] != sizes[2]:
        C = tensor.ndim - 1
    elif sizes[1] == sizes[2] and sizes[0] != sizes[1]:
        C = tensor.ndim - 3
    # all sizes are different in the case below
    elif len(possible_channel_dims) == 1:
        C = possible_channel_dims[0]
    elif len(possible_alpha_channel_dims) == 1:
        C = possible_alpha_channel_dims[0]
    elif ignore_error:
        return tensor
    else:
        raise ValueError("Can not find exact channel dim.")

    permute = list(range(tensor.ndim))
    permute.pop(C)
    if target_format.endswith("last"):
        permute.append(C)
    else:
        permute.insert(tensor.ndim - 3, C)
    return tensor.permute(permute)


def tensor_to_image(tensor: Tensor, auto_permute: bool = False) -> ndarray:
    """Return image ndarray from tensor."""
    if auto_permute:
        tensor = image_auto_permute(
            tensor, target_format="NHWC", ignore_error=True
        )
    tensor = normalize_image_array(tensor.detach().cpu(), 0, 255)
    return tensor.add_(0.5).clamp_(0, 255).byte().numpy()


def save_tensor_to_image(tensor: Tensor, uri: Any) -> Optional[bytes]:
    """Save tensor to image."""
    image = tensor_to_image(tensor, auto_permute=True)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    if not isinstance(uri, str):
        if image.ndim != 3:
            raise ValueError(
                "Only support single image when uri is not string."
            )
        return imageio.imwrite(uri, image)

    path, ext = os.path.splitext(uri)
    if not ext:
        ext = ".png"

    if image.ndim == 3:
        return imageio.imwrite(path + ext, image)
    for i in range(image.shape[0]):
        imageio.imwrite(path + f"_{i:03d}_{ext}", image[i])


@overload
def normalize_image_array(
    array: Tensor, min: float = 0.0, max: float = 1.0
) -> Tensor:
    ...


@overload
def normalize_image_array(
    array: ndarray, min: float = 0.0, max: float = 1.0
) -> ndarray:
    ...


def normalize_image_array(
    array: Union[Tensor, ndarray], min: float = 0.0, max: float = 1.0
) -> Union[Tensor, ndarray]:
    """
    Return a new min-max normalized image from following 3 cases:
        case 1: [0, 255]
        case 2: [0, 1]
        case 3: [-1, 1]

    Input array should be in range of the cases above.
    Should clamp the values of input array before normalization.
    """
    # If is int array, cast to float32.
    if isinstance(array, ndarray) and array.dtype != np.float32:
        array = array.astype(np.float32)
    elif isinstance(array, Tensor) and array.dtype != torch.float32:
        array = array.float()

    # normalize to [0, 1]
    array_min, array_max = array.min(), array.max()
    if 0.0 <= array_min and array_max <= 255.0:
        if 1.0 < array_max:  # case 1: 0.0 <= array <= 255.0
            array = array / 255.0
        else:  # case 2: 0.0 <= array <= 1.0
            pass  # Do nothing.
    # case 3: -1.0 <= array <= 1.0
    elif -1.0 <= array_min and array_max <= 1.0:
        array = array * 0.5 + 0.5
    else:  # Otherwise, scaling by it's min, max values.
        array = (array - array_min) / (array_max - array_min)

    # min-max scaling.
    return array * (max - min) + min


@overload
def batch_flat_to_square(
    array: Tensor, channels: Optional[int] = None
) -> Tensor:
    ...


@overload
def batch_flat_to_square(
    array: ndarray, channels: Optional[int] = None
) -> ndarray:
    ...


def batch_flat_to_square(
    array: Union[Tensor, ndarray], channels: Optional[int] = None
) -> Union[Tensor, ndarray]:
    """
    Return a (batch, channels, height, width) shaped square array
    from (batch, features) shaped array.
    """
    N, F = array.shape[:2]
    if channels is None:
        if squared_number(F):
            channels = 1
        elif squared_number(F / 3):
            channels = 3
        else:
            raise ValueError(
                f"Can not infer channels from array shape: {array.shape}"
            )
    resolution = squared_number(F // channels)
    return array.reshape(N, channels, resolution, resolution)


def match_channels_between_NCHW_tensor(
    image_a: Tensor, image_b: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Match channels to each other with a larger number of channels.
    """
    if image_a.shape[2:] != image_b.shape[2:]:
        raise ValueError(
            "Image resolution mismatch: "
            f"{image_a.shape[2:]} != {image_b.shape[2:]}"
        )
    C_a = image_a.shape[1]
    C_b = image_b.shape[1]
    if (
        C_a != C_b
        and not (
            (C_a in {3, 4} and C_b in {1, 3, 4})
            or ((C_a in {1, 3, 4} and C_b in {3, 4}))
        )
        and C_a != 1
        and C_b != 1
    ):
        raise ValueError(
            f"Can not match channels between {C_a} and {C_b} channels."
        )

    if C_a == C_b:
        return image_a, image_b

    swap = C_a > C_b
    if swap:
        image_a, image_b = image_b, image_a
        C_a, C_b = C_b, C_a

    if C_b == 4:
        if C_a == 1:
            image_a = image_a.repeat(1, 3, 1, 1)
        size = list(image_a.shape)
        size[1] = 1
        ones = torch.ones(size, dtype=image_a.dtype, device=image_a.device)
        image_a = torch.cat([image_a, ones], dim=1)
    else:
        image_a = image_a.repeat(1, C_b, 1, 1)

    if swap:
        return image_b, image_a
    return image_a, image_b


def get_semantic_one_hot(
    input: Tensor,
    n_classes: int,
    normalize: bool = False,
    input_range: Tuple[float, float] = (-1.0, 1.0),
) -> Tensor:
    """
    Converts a tensor of indices to a one-hot tensor.

    Args:
        input: (N, 1, H, W) semantic label, range [0, n_classes - 1]
        n_classes: The number of classes.
        normalize: If True, normalize input to [0, n_classes - 1].
        input_range: The range of input.

    Returns:
        A tensor of one-hot vectors. (N, n_classes, H, W)
    """
    if normalize:
        min, max = input_range
        input = (input.clamp_(min, max) - min) / (max - min) * (n_classes - 1)
    one_hot_size = list(input.shape)
    one_hot_size[1] = n_classes
    one_hot = torch.zeros(one_hot_size, dtype=input.dtype, device=input.device)
    return one_hot.scatter_(1, input.long(), 1)


def get_semantic_edge(input: Tensor) -> Tensor:
    """
    Args:
        input: (N, 1, H, W) semantic label, range Any

    Returns:
        A tensor of edge mask. (N, 1, H, W), contains 0 or 1
    """
    difference_along_width = input[..., 1:] != input[..., :-1]
    difference_along_height = input[..., 1:, :] != input[..., :-1, :]

    edge = torch.zeros_like(input, dtype=torch.bool)
    edge[..., 1:] |= difference_along_width
    edge[..., :-1] |= difference_along_width
    edge[..., 1:, :] |= difference_along_height
    edge[..., :-1, :] |= difference_along_height
    return edge.to(dtype=input.dtype)


def tensor_resize(
    image: Tensor,
    size: INT,
    interpolation: str = "bilinear",
    antialias: Optional[bool] = None,
) -> Tensor:
    """
    Resize a tensor of images to given `size`.

    Args:
        image: Tensor of shape (C, H, W) or (N, C, H, W).
        size: Desired output size.
            If `size` is an integer, larger edge of the image will be matched to
            this number preserving the aspect ratio.
            If `size` is a sequence of two integers like (h, w), output size
            will be matched to this sequence.
        interpolation: Interpolation mode to calculate output values.
            Only {"nearest", "bilinear", "bicubic"} are supported.
        antialias: Whether to apply anti-aliasing filter.
    """
    interpolation = interpolation.lower()
    if isinstance(size, int):
        short, long = sorted(image.shape[-2:])
        if short == long:
            return tensor_resize(image, (size, size), interpolation, antialias)
        if long == size:
            return image
        size = (round(short * size / long), size)
    elif isinstance(size, Sequence):
        size = normalize_int_tuple(size, 2)
    else:
        raise TypeError(f"Unexpected size type: {type(size)}")
    if image.size(-1) == size[-1] and image.size(-2) == size[-2]:
        return image
    if image.device.type == "cpu" or antialias != True:
        return TFT.resize(image, size, interpolation, antialias=antialias)

    need_twice = False
    temp_size = []
    for target, original in zip(size, image.shape[2:]):
        ratio = float(target) / float(original)
        # If antialias is True and ratio is less than 0.01, need to resize twice
        # to avoid `RuntimeError: Too much shared memory required`
        if ratio < 0.01:
            need_twice = True
            temp_size.append(target * 10)
        else:
            temp_size.append(target)
    if need_twice:
        image = TFT.resize(image, temp_size, interpolation, antialias=antialias)
    return TFT.resize(image, size, interpolation, antialias=antialias)
