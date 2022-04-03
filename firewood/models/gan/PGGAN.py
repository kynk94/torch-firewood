import argparse
import functools
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional_tensor as TFT
from torch import Tensor

from firewood import layers
from firewood.common.types import INT
from firewood.layers import activations
from firewood.layers.upfirdn import upsample


class ToImage(layers.GFixConv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        bias: bool = True,
        activation: str = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activations.get(activation)

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def extra_repr(self) -> str:
        repr = super().extra_repr()
        if self.activation is not None:
            repr += f", activation={self.activation}"
        return repr


class Upsample(nn.Module):
    def __init__(
        self,
        size: Optional[INT] = None,
        factor: Optional[int] = None,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        if not ((size is None) ^ (factor is None)):
            raise ValueError("Exactly one of size or factor must be specified")
        self.size = size
        self.factor = factor
        self.mode = mode

        if self.mode in {"zeros", "nearest"}:
            self.upsample = functools.partial(
                upsample, factor=self.factor, mode=self.mode
            )
        else:
            self.upsample = functools.partial(
                TFT.resize,
                size=self.size,
                interpolation=self.mode,
                antialias=True,
            )

    def forward(self, input: Tensor) -> Tensor:
        return self.upsample(input)


class Generator(nn.Module):
    """
    Generator of PGGAN
    Progressive Growing of GANs for Improved Quality, Stability, and Variation
    https://arxiv.org/abs/1710.10196
    """

    def __init__(
        self,
        latent_dim: int = 512,
        max_filters: int = 512,
        init_resolution: int = 4,
        out_channels: int = 3,
        padding_mode: str = "reflect",
        activation: str = "lrelu",
        normalize_latent: bool = True,
        freeze_lower_layers: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.max_filters = max_filters
        self.init_resolution = init_resolution
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.activation = activation
        self.normalize_latent = normalize_latent
        self.freeze_lower_layers = freeze_lower_layers

        block = []
        if self.normalize_latent:
            block.append(layers.PixelNorm())
        # fmt: off
        block.extend([
            layers.LinearBlock(latent_dim, self.max_filters * init_resolution**2, bias=True,
                               op_order="WAN", activation=activation, normalization="pixel"),
            layers.Reshape(self.max_filters, init_resolution, init_resolution),
            layers.Conv2dBlock(self.max_filters, self.n_filters(init_resolution), 3, 1, "same", bias=True,
                               padding_mode=self.padding_mode, op_order="WAN", normalization="pixel", activation=activation),
        ])
        # fmt: on
        self.blocks = nn.ModuleList([nn.Sequential(*block)])
        self.to_image = nn.ModuleList([ToImage(latent_dim, self.out_channels)])
        self.out_resolution = self.init_resolution

    def forward(self, input: Tensor, alpha: float = 1.0) -> Tensor:
        output = input
        features = []
        for block in self.blocks:
            output = block(output)
            resolution = output.shape[-1]
            if resolution >= self.out_resolution // 2:
                features.append(output)
        image = self.to_image[-1](features.pop())
        if alpha != 1.0 and features:
            upsampled_feature = upsample(
                features.pop(), factor=2, mode="nearest"
            )
            lower_image = self.to_image[-2](upsampled_feature)
            image = alpha * image + (1 - alpha) * lower_image
        return image

    @torch.no_grad()
    def inference_all(
        self, input: Tensor, alpha: float = 1.0
    ) -> Tuple[Tensor, ...]:
        self.eval()

        output = input
        images = []
        lower_images = [None]
        for block, to_image in zip(self.blocks, self.to_image):
            output = block(output)
            images.append(to_image(output))
            if alpha != 1.0 and output.shape[-1] < self.out_resolution:
                upsampled_feature = upsample(output, factor=2, mode="nearest")
                lower_images.append(to_image(upsampled_feature))

        for i in range(1, len(lower_images)):
            images[i] = alpha * images[i] + (1 - alpha) * lower_images[i]

        self.train()
        return tuple(images)

    def n_filters(self, out_resolution: int) -> int:
        return min(self.max_filters, 2**14 // out_resolution)

    def grow_resolution(self, resolution: int) -> None:
        if resolution < self.out_resolution:
            raise ValueError(
                "Resolution must be greater than current resolution"
            )
        self.out_resolution = resolution

        if self.freeze_lower_layers:
            for param in self.parameters():
                param.requires_grad = False

        # fmt: off
        in_channels = self.n_filters(self.out_resolution // 2)
        out_channels = self.n_filters(self.out_resolution)
        conv_kwargs = dict(kernel_size=3, stride=1, padding="same", bias=True,
                           padding_mode="reflect", op_order="WAN", normalization="pixel", activation=self.activation)
        block = nn.Sequential(
            Upsample(factor=2, mode="nearest"),
            layers.Conv2dBlock(in_channels, out_channels, **conv_kwargs),
            layers.Conv2dBlock(out_channels, out_channels, **conv_kwargs),
        )
        self.blocks.append(block)
        self.to_image.append(ToImage(out_channels, self.out_channels))
        # fmt: on


class Discriminator(nn.Module):
    """
    Discriminator of PGGAN
    Progressive Growing of GANs for Improved Quality, Stability, and Variation
    https://arxiv.org/abs/1710.10196
    """

    def __init__(self):
        super().__init__()


def main() -> None:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.model_summary import summarize

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", "-d", type=int, default=3)
    args = vars(parser.parse_args())

    class Summary(pl.LightningModule):
        def __init__(
            self,
            latent_dim: int = 512,
            init_resolution: int = 4,
            out_channels: int = 3,
            padding_mode: str = "reflect",
            activation: str = "lrelu",
            normalize_latent: bool = True,
            freeze_lower_layers: bool = True,
        ) -> None:
            super().__init__()
            self.generator = Generator(
                latent_dim=latent_dim,
                init_resolution=init_resolution,
                out_channels=out_channels,
                padding_mode=padding_mode,
                activation=activation,
                normalize_latent=normalize_latent,
                freeze_lower_layers=freeze_lower_layers,
            )
            for i in range(3, 11):
                self.generator.grow_resolution(2**i)
            self.example_input_array = (torch.empty(2, latent_dim), 0.5)

        def forward(self, input: Tensor, alpha: float = 1.0) -> Tensor:  # type: ignore
            generated_images = self.generator.inference_all(input, alpha)
            return generated_images[0]

    summary = Summary()
    print(summary)
    print(summarize(summary, max_depth=args["depth"]))


if __name__ == "__main__":
    main()
