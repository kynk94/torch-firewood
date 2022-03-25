import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

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
        compute_on_cpu: bool = True,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable[[Tensor], List[Tensor]] = None,
    ) -> None:
        super().__init__(
            feature=feature,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.inception.eval()
        self.inception_resolution: int = getattr(
            self.inception, "INPUT_IMAGE_SIZE", 299
        )
        self.resize_lib = _to_resize_lib(resize_lib)
        self.compute_on_cpu = compute_on_cpu
        if self.compute_on_cpu:
            self._to_sync = False

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
        if self.compute_on_cpu:
            features = features.detach().cpu()

        if is_real:
            self.real_features.append(features)
        else:
            self.fake_features.append(features)

    def compute(self) -> Tensor:
        output = super().compute()
        return output.to(device=self.device, non_blocking=True)
