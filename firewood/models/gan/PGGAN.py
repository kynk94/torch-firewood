import argparse
import functools
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional_tensor as TFT
from torch import Tensor

from firewood import layers
from firewood.common.types import INT
from firewood.layers.upfirdn import nearest_downsample, upsample


class ToImage(layers.Conv2dBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        bias: bool = True,
        activation: Optional[str] = None,
        lr_equalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            activation=activation,
            lr_equalization=lr_equalization,
            **kwargs,
        )


class FromImage(layers.Conv2dBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        normalization: Optional[str] = None,
        activation: str = "lrelu",
        lr_equalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            normalization=normalization,
            activation=activation,
            lr_equalization=lr_equalization,
            **kwargs,
        )


class Upsample(nn.Module):
    def __init__(
        self,
        factor: Optional[int] = None,
        size: Optional[INT] = None,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        if not ((factor is None) ^ (size is None)):
            raise ValueError("Exactly one of factor or size must be specified")
        self.factor = factor
        self.size = size
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


class Downsample(nn.Module):
    def __init__(
        self,
        factor: Optional[int] = None,
        size: Optional[INT] = None,
        mode: str = "nearest",
    ) -> None:
        super().__init__()
        if not ((factor is None) ^ (size is None)):
            raise ValueError("Exactly one of factor or size must be specified")
        self.factor = factor
        self.size = size
        self.mode = mode

        if self.mode == "nearest":
            self.downsample = functools.partial(
                nearest_downsample, factor=self.factor
            )
        else:
            self.downsample = functools.partial(
                TFT.resize,
                size=self.size,
                interpolation=self.mode,
                antialias=True,
            )

    def forward(self, input: Tensor) -> Tensor:
        return self.downsample(input)


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
        use_tanh: bool = False,
        normalize_latent: bool = True,
        freeze_lower_layers: bool = False,
        lr_equalization: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.max_filters = max_filters
        self.init_resolution = init_resolution
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.activation = activation
        self.use_tanh = use_tanh
        self.normalize_latent = normalize_latent
        self.freeze_lower_layers = freeze_lower_layers
        self.lr_equalization = lr_equalization

        block = []
        if self.normalize_latent:
            block.append(layers.PixelNorm())
        # fmt: off
        kwargs = dict(bias=True, op_order="WAN", normalization="pixel",
                      activation=self.activation, lr_equalization=self.lr_equalization)
        block.extend([
            layers.LinearBlock(self.latent_dim, self.max_filters * self.init_resolution**2, **kwargs),
            layers.Reshape(self.max_filters, self.init_resolution, self.init_resolution),
            layers.Conv2dBlock(self.max_filters, self.n_filters(self.init_resolution), 3, 1, "same",
                               padding_mode=self.padding_mode, **kwargs),
        ])
        self.blocks = nn.ModuleList([nn.Sequential(*block)])
        self.to_image = nn.ModuleList([
            ToImage(self.latent_dim, self.out_channels, bias=True,
                    activation="tanh" if self.use_tanh else None, lr_equalization=self.lr_equalization)
        ])
        # fmt: on
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

    def n_filters(self, resolution: int) -> int:
        return min(self.max_filters, 2**14 // resolution)

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
                           padding_mode=self.padding_mode, op_order="WAN",
                           normalization="pixel", activation=self.activation, lr_equalization=self.lr_equalization)
        block = nn.Sequential(
            Upsample(factor=2, mode="nearest"),
            layers.Conv2dBlock(in_channels, out_channels, **conv_kwargs),
            layers.Conv2dBlock(out_channels, out_channels, **conv_kwargs),
        )
        self.blocks.append(block)
        self.to_image.append(ToImage(out_channels, self.out_channels, lr_equalization=self.lr_equalization))
        # fmt: on


class Discriminator(nn.Module):
    """
    Discriminator of PGGAN
    Progressive Growing of GANs for Improved Quality, Stability, and Variation
    https://arxiv.org/abs/1710.10196
    """

    def __init__(
        self,
        in_channels: int = 3,
        max_filters: int = 512,
        init_resolution: int = 4,
        minibatch_size: int = 4,
        label_dim: int = 0,
        padding_mode: str = "reflect",
        normalization: Optional[str] = None,
        activation: str = "lrelu",
        freeze_lower_layers: bool = False,
        lr_equalization: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.max_filters = max_filters
        self.init_resolution = init_resolution
        self.minibatch_size = minibatch_size
        self.label_dim = label_dim
        self.padding_mode = padding_mode
        self.normalization = normalization
        self.activation = activation
        self.freeze_lower_layers = freeze_lower_layers
        self.lr_equalization = lr_equalization

        # fmt: off
        out_channels = self.n_filters(self.init_resolution)
        self.from_image = nn.ModuleList([
            FromImage(self.in_channels, out_channels, bias=True,
                      normalization=self.normalization, activation=self.activation,
                      lr_equalization=self.lr_equalization)
        ])

        kwargs = dict(bias=True, normalization=self.normalization, activation=self.activation,
                      lr_equalization=self.lr_equalization)
        block = nn.Sequential(
            layers.MinibatchStd(size=minibatch_size),
            layers.Conv2dBlock(out_channels + 1, out_channels, 3, 1, "same",
                               padding_mode=self.padding_mode, **kwargs),
            layers.Conv2dBlock(out_channels, out_channels, self.init_resolution, 1, 0, **kwargs),
            layers.LinearBlock(out_channels, 1 + self.label_dim, bias=True, lr_equalization=self.lr_equalization),
        )
        self.blocks = nn.ModuleList([block])
        # fmt: on
        self.in_resolution = self.init_resolution

    def forward(self, input: Tensor, alpha: float = 1.0) -> Tensor:
        feature = self.from_image[0](input)
        output = self.blocks[0](feature)
        if alpha != 1.0 and self.in_resolution > self.init_resolution:
            downsampled_input = nearest_downsample(input, factor=2)
            lower_feature = self.from_image[1](downsampled_input)
            output = alpha * output + (1 - alpha) * lower_feature

        for block in self.blocks[1:]:
            output = block(output)
        return output

    def n_filters(self, resolution: int) -> int:
        return min(self.max_filters, 2**14 // resolution)

    def grow_resolution(self, resolution: int) -> None:
        if resolution < self.in_resolution:
            raise ValueError(
                "Resolution must be greater than current resolution"
            )
        self.in_resolution = resolution

        if self.freeze_lower_layers:
            for param in self.parameters():
                param.requires_grad = False

        # fmt: off
        in_channels = self.n_filters(self.in_resolution)
        out_channels = self.n_filters(self.in_resolution // 2)
        from_image = FromImage(self.in_channels, in_channels, bias=True,
                               normalization=self.normalization, activation=self.activation,
                               lr_equalization=self.lr_equalization)
        self.from_image = nn.ModuleList([from_image, *self.from_image])

        conv_kwargs = dict(kernel_size=3, stride=1, padding="same", bias=True,
                           padding_mode=self.padding_mode, normalization=self.normalization,
                           activation=self.activation, lr_equalization=self.lr_equalization)
        block = nn.Sequential(
            layers.Conv2dBlock(in_channels, in_channels, **conv_kwargs),
            layers.Conv2dBlock(in_channels, out_channels, **conv_kwargs),
            Downsample(factor=2, mode="nearest"),
        )
        self.blocks = nn.ModuleList([block, *self.blocks])
        # fmt: on


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
            lr_equalization: bool = True,
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
                lr_equalization=lr_equalization,
            )
            self.discriminator = Discriminator(
                in_channels=out_channels,
                init_resolution=init_resolution,
                minibatch_size=2,
                label_dim=0,
                padding_mode=padding_mode,
                normalization=None,
                activation=activation,
                freeze_lower_layers=freeze_lower_layers,
                lr_equalization=lr_equalization,
            )
            for i in range(3, 11):
                self.generator.grow_resolution(2**i)
                self.discriminator.grow_resolution(2**i)
            self.example_input_array = (torch.empty(4, latent_dim), 0.5)

        def forward(self, input: Tensor, alpha: float = 1.0) -> Tensor:  # type: ignore
            _ = self.generator.inference_all(input, alpha)
            generated_image = self.generator(input, alpha)
            score = self.discriminator(generated_image, alpha)
            return score

    summary = Summary()
    print(summary)
    print(summarize(summary, max_depth=args["depth"]))


if __name__ == "__main__":
    main()
