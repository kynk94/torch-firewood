import warnings
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional_tensor as TFT
from PIL import Image
from torch import Tensor
from torch.nn.modules.utils import _pair
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from firewood.common.types import INT


class RESIZE_LIB(Enum):
    TORCH = "torch"
    PIL = "pil"
    TF = "tf"


def _to_resize_lib(resize_lib: Union[str, RESIZE_LIB]) -> RESIZE_LIB:
    if isinstance(resize_lib, RESIZE_LIB):
        return resize_lib
    resize_lib = resize_lib.strip().lower()
    if resize_lib.startswith("torch"):
        return RESIZE_LIB.TORCH
    if resize_lib.startswith("pil"):
        return RESIZE_LIB.PIL
    if resize_lib.startswith("tf"):
        return RESIZE_LIB.TF
    raise ValueError(f"Unknown resize library: {resize_lib}")


class FrechetInceptionDistance(FID):
    def __init__(
        self,
        feature: Union[int, nn.Module] = 2048,
        resize_lib: Union[str, RESIZE_LIB] = RESIZE_LIB.TORCH,
        reset_real_features: bool = True,
        compute_on_cpu: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.update({"compute_on_cpu": compute_on_cpu})
        super().__init__(
            feature=feature,
            reset_real_features=reset_real_features,
            **kwargs,
        )
        self.inception.eval()
        self.inception_resolution: int = getattr(
            self.inception, "INPUT_IMAGE_SIZE", 299
        )
        self.resize_lib = _to_resize_lib(resize_lib)

    def resize(self, images: Tensor, size: Optional[INT] = None) -> Tensor:
        if self.resize_lib == RESIZE_LIB.TF:
            # tf1 resize automatically in inception module.
            return images

        target_resolution = _pair(size or self.inception_resolution)
        device = images.device
        if self.resize_lib == RESIZE_LIB.TORCH:
            return TFT.resize(
                img=images,
                size=target_resolution,
                interpolation="bilinear",
                antialias=True,
            )
        if self.resize_lib == RESIZE_LIB.PIL:
            # Not recommended to use.
            # PIL is over 10x slower than others, but good for fid.
            resized_images = []
            for image in images.detach().cpu().numpy():
                channels = []
                for channel in image:
                    resized_channel = Image.fromarray(channel, mode="F").resize(
                        size=target_resolution, resample=Image.BILINEAR
                    )
                    channels.append(np.array(resized_channel))
                resized_images.append(np.stack(channels, axis=0))
            resized_tensor = torch.from_numpy(np.stack(resized_images, axis=0))
            return resized_tensor.to(device=device)
        raise ValueError(f"Not supported resize library: {self.resize_lib}")

    @torch.no_grad()
    # type: ignore[override]
    def update(
        self,
        images: Tensor,
        is_real: bool,
        normalize: bool = True,
        images_range: Tuple[int, int] = (-1, 1),
    ) -> None:
        """
        images: image tensor with shape (N, C, H, W)
            if normalize is False, images should be in range of (0, 255).
            else, images should be in images_range.
        """
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.size(1) > 3:
            warnings.warn("Images with more than 3 channels are not supported.")
            images = images[:, :3, :, :]

        if normalize:
            images = (images - images_range[0]) / (
                images_range[1] - images_range[0]
            ) * 255 + 0.5
        if self.resize_lib != "tf1":
            images = self.resize(images)
        features: Tensor = self.inception(images.clamp_(0, 255).byte())

        if is_real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)
