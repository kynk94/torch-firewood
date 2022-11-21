import argparse
import math
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from firewood import hooks, layers, utils
from firewood.common.types import NUMBER
from firewood.layers.biased_activations import ACTIVATIONS


class MappingNetwork(nn.Module):
    """
    Mapping Network of StyleGAN

    latent_dim (latent z)
        |
    hidden_dim * n_layers (mlp)
        |
    style_dim (style w)
    """

    style_avg: Tensor

    def __init__(
        self,
        latent_dim: int = 512,
        label_dim: int = 0,
        hidden_dim: int = 512,
        style_dim: int = 512,
        n_layers: int = 8,
        normalize_latents: bool = True,
        bias: bool = True,
        activation: str = "lrelu",
        style_avg_beta: float = 0.995,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.style_avg_beta = style_avg_beta
        self.register_buffer("style_avg", torch.zeros(style_dim))

        # conditional generation
        if self.label_dim == 0:
            self.label_affine = None
            in_features = self.latent_dim
        else:
            self.label_affine = layers.Linear(
                label_dim, self.latent_dim, bias=False
            )
            in_features = self.latent_dim * 2

        self.layers = nn.ModuleList()

        # normalize latent vector
        if normalize_latents:
            self.layers.append(layers.PixelNorm())

        # mapping network
        layer_kwargs = dict(bias=bias, activation=activation)
        for i in range(n_layers):
            if i == 0:  # first layer
                out_features = self.hidden_dim
            elif i < n_layers - 1:
                in_features = out_features
            else:  # last layer
                out_features = self.style_dim
            self.layers.append(
                layers.LinearBlock(in_features, out_features, **layer_kwargs)
            )

    def forward(self, input: Tensor, label: Optional[Tensor] = None) -> Tensor:
        if self.label_affine is not None:
            label = self.label_affine(label)
            input = torch.cat([input, label], dim=1)

        output = input
        for layer in self.layers:
            output = layer(output)
        if self.training and self.style_avg_beta is not None:
            with torch.no_grad():
                self.style_avg.lerp_(
                    output.float().mean(0), 1 - self.style_avg_beta
                )
        return output


# ------------------------------------------------------------------------------


class InitialBlock(nn.Module):
    def __init__(
        self,
        style_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[str, int] = "same",
        initial_resolution: int = 4,
        bias: bool = True,
        activation: str = "lrelu",
        noise: Optional[str] = "normal",
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_resolution = initial_resolution
        self.input = nn.Parameter(
            torch.ones(
                self.in_channels,
                self.initial_resolution,
                self.initial_resolution,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.in_channels))
        else:
            self.register_buffer("bias", None)
        self.modulated_conv = layers.Conv2dBlock(
            self.in_channels,
            self.out_channels,
            kernel_size,
            stride,
            padding,
            bias=bias,
            activation=activation,
            noise=noise,
        )
        hooks.weight_denorm(self.modulated_conv, self.style_dim)

    def forward(self, input: Tensor) -> Tensor:
        """
        input: style vector (w)
        """
        if input.ndim == 3:
            input = input[:, 0]
        output = self.input.repeat(input.size(0), 1, 1, 1)
        output = self.modulated_conv(output, input)
        return output


class ToImage(nn.Module):
    def __init__(
        self,
        style_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.modulated_conv = layers.GFixConv2d(
            in_channels, out_channels, kernel_size, bias=bias
        )
        hooks.weight_denorm(
            self.modulated_conv, self.style_dim, demodulate=False
        )

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        """
        input: feature map
        style: style vector (w)
        """
        if style.ndim == 3:
            style = style[:, 0]
        output = self.modulated_conv(input, style)
        return output


class UpConvConvBlock(nn.Module):
    def __init__(
        self,
        style_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[str, int] = "same",
        bias: bool = True,
        activation: str = "lrelu",
        noise: Optional[str] = "normal",
        fir: NUMBER = [1, 3, 3, 1],
        up: int = 2,
    ) -> None:
        super().__init__()
        kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            activation=activation,
            noise=noise,
            weight_normalization="denorm",
            weight_normalization_args=dict(modulation_features=style_dim),
        )
        self.modulated_conv_1 = layers.Conv2dBlock(
            in_channels, out_channels, fir=fir, up=up, **kwargs
        )
        self.modulated_conv_2 = layers.Conv2dBlock(
            out_channels, out_channels, **kwargs
        )

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        if style.ndim == 2:
            style = style.unsqueeze(1).repeat(1, 2, 1)
        output = self.modulated_conv_1(input, style[:, 0])
        output = self.modulated_conv_2(output, style[:, 1])
        return output


class SynthesisNetwork(nn.Module):
    initial_resolution = 4

    def __init__(
        self,
        style_dim: int = 512,
        max_channels: int = 512,
        channels_decay: float = 1.0,
        activation: str = "lrelu",
        noise: Optional[str] = "normal",
        fir: NUMBER = [1, 3, 3, 1],
        resolution: int = 1024,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.max_channels = max_channels
        self.channels_decay = channels_decay
        self.resolution = self._check_resolution(resolution)
        self.image_channels = image_channels
        self.n_layers = (
            int(math.log2(self.resolution / self.initial_resolution)) + 1
        )

        self.layers = nn.ModuleDict()
        self.to_images = nn.ModuleDict()
        conv_kwargs = dict(
            kernel_size=3,
            stride=1,
            padding="same",
            bias=True,
            activation=activation,
            noise=noise,
        )
        for i in range(self.n_layers):
            resolution = 2 ** (i + 2)
            if i == 0:
                in_channels = _calc_channels(
                    resolution=self.initial_resolution,
                    max=self.max_channels,
                    decay=self.channels_decay,
                )
                out_channels = in_channels
                self.layers[str(resolution)] = InitialBlock(
                    style_dim=self.style_dim,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    initial_resolution=self.initial_resolution,
                    **conv_kwargs,
                )
            else:
                in_channels = out_channels
                out_channels = _calc_channels(
                    resolution=resolution,
                    max=self.max_channels,
                    decay=self.channels_decay,
                )
                self.layers[str(resolution)] = UpConvConvBlock(
                    style_dim,
                    in_channels,
                    out_channels,
                    fir=fir,
                    up=2,
                    **conv_kwargs,
                )

            self.to_images[str(resolution)] = ToImage(
                style_dim, out_channels, self.image_channels, 1, bias=True
            )

        self.upfir = layers.UpFirDn2d(fir, up=2)

        self.style_indices = []
        for i in range(self.n_layers):
            self.style_indices.append(list(range(max(i * 2 - 1, 0), i * 2 + 2)))

    def forward(
        self,
        input: Tensor,
        resolution: Optional[int] = None,
    ) -> Tensor:
        """
        Args:
            input: style vector (w)
            resolution: resolution of output image
        """
        if resolution is None:
            resolution = self.resolution

        if input.ndim == 2:
            input = input.unsqueeze(1).repeat(1, self.n_layers * 2, 1)

        prev_image: Optional[Tensor] = None
        for index, (name, layer) in enumerate(self.layers.items()):
            *layer_style_index, image_style_index = self.style_indices[index]
            layer_style = input[:, layer_style_index]
            image_style = input[:, image_style_index]
            if name == str(self.initial_resolution):
                output = layer(layer_style)
            else:
                output = layer(output, layer_style)
            image = self.to_images[name](output, image_style)
            if prev_image is not None:
                image = image + self.upfir(prev_image)
            prev_image = image
            if name == str(resolution):
                break
        return image

    def _check_resolution(self, resolution: int) -> int:
        if resolution < self.initial_resolution:
            raise ValueError(
                f"`resolution` must be >= {self.initial_resolution}"
            )
        if resolution != utils.highest_power_of_2(resolution):
            raise ValueError(
                f"`resolution` must be a power of 2, but got {resolution}"
            )
        return resolution


# ------------------------------------------------------------------------------


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        label_dim: int = 0,
        style_dim: int = 512,
        max_channels: int = 512,
        activation: str = "lrelu",
        noise: Optional[str] = "normal",
        fir: NUMBER = [1, 3, 3, 1],
        resolution: int = 1024,
        image_channels: int = 3,
        truncation_cutoff: int = 8,
        style_mixing_probability: Optional[float] = 0.9,
        lr_equalization: bool = True,
        mapping_lr_multiplier: float = 0.01,
        synthesis_lr_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.mapping = MappingNetwork(
            latent_dim=latent_dim,
            label_dim=label_dim,
            style_dim=style_dim,
            activation=activation,
        )
        self.synthesis = SynthesisNetwork(
            style_dim=style_dim,
            max_channels=max_channels,
            activation=activation,
            noise=noise,
            fir=fir,
            resolution=resolution,
            image_channels=image_channels,
        )
        if lr_equalization:
            hooks.lr_equalizer(
                module=self.mapping, lr_multiplier=mapping_lr_multiplier
            )
            hooks.lr_equalizer(
                module=self.synthesis, lr_multiplier=synthesis_lr_multiplier
            )

        self.truncation_cutoff = truncation_cutoff
        self.style_mixing_probability = utils.clamp(
            style_mixing_probability, 0.0, 1.0
        )

    def forward(
        self,
        input: Tensor,
        label: Optional[Tensor] = None,
        truncation: Optional[float] = 0.5,
        resolution: Optional[int] = None,
    ) -> Tensor:
        if resolution is None:
            resolution = self.synthesis.resolution

        style: Tensor = self.mapping(input, label)
        style = style.unsqueeze(1).repeat(1, self.synthesis.n_layers * 2, 1)

        layer_index = torch.arange(
            self.synthesis.n_layers * 2, device=input.device
        ).view(1, -1, 1)
        if (
            self.training
            and self.style_mixing_probability is not None
            and random.random() < self.style_mixing_probability
        ):
            mixing_input = torch.randn_like(input)
            mixing_style: Tensor = self.mapping(mixing_input, label)
            mixing_style = mixing_style.unsqueeze(1).repeat(
                1, self.synthesis.n_layers * 2, 1
            )
            current_index = 2 + 2 * math.log2(
                resolution / self.synthesis.initial_resolution
            )
            mixing_cutoff = random.randint(1, int(current_index))
            style = torch.where(
                layer_index < mixing_cutoff, style, mixing_style
            )
        if truncation is not None and self.truncation_cutoff is not None:
            truncation = utils.clamp(truncation, 0.0, 1.0)
            style = torch.where(
                layer_index < self.truncation_cutoff,
                self.mapping.style_avg.lerp(style.float(), truncation),
                style,
            )
        image = self.synthesis(style, resolution)
        return image


# ------------------------------------------------------------------------------


class ConvConvDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[str, int] = "same",
        bias: bool = True,
        activation: str = "lrelu",
        fir: NUMBER = [1, 3, 3, 1],
        down: int = 2,
    ) -> None:
        super().__init__()
        kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            activation=activation,
        )
        self.conv_1 = layers.Conv2dBlock(in_channels, in_channels, **kwargs)
        self.conv_2 = layers.Conv2dBlock(
            in_channels,
            out_channels,
            fir=fir,
            down=down,
            **kwargs,
        )
        self.conv_skip = layers.Conv2dBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            fir=fir,
            down=down,
            bias=False,
        )
        self.activation_gain = 1 / math.sqrt(2)

    def forward(self, input: Tensor) -> Tensor:
        skip = self.conv_skip(input)
        output = self.conv_1(input)
        output = self.conv_2(output)
        return self.activation_gain * (output + skip)


class Discriminator(nn.Module):
    initial_resolution = 4

    def __init__(
        self,
        label_dim: int = 0,
        max_channels: int = 512,
        channels_decay: float = 1.0,
        mbstd_group: int = 4,
        activation: str = "lrelu",
        fir: NUMBER = [1, 2, 1],
        resolution: int = 1024,
        image_channels: int = 3,
        lr_equalization: bool = True,
        lr_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.label_dim = label_dim
        self.max_channels = max_channels
        self.channels_decay = channels_decay
        self.mbstd_group = mbstd_group
        self.resolution = resolution
        self.image_channels = image_channels
        self.n_layers = (
            int(math.log2(self.resolution / self.initial_resolution)) + 1
        )

        self.layers = nn.ModuleDict()
        self.from_images = nn.ModuleDict()
        kwargs = dict(bias=True, activation=activation)
        for i in range(self.n_layers - 1, -1, -1):
            resolution = 2 ** (i + 2)
            if i == self.n_layers - 1:
                in_channels = _calc_channels(
                    resolution, self.max_channels, self.channels_decay
                )
            else:
                in_channels = out_channels
            out_channels = _calc_channels(
                resolution // 2, self.max_channels, self.channels_decay
            )

            self.from_images[str(resolution)] = layers.Conv2dBlock(
                self.image_channels, in_channels, 1, 1, 0, **kwargs
            )
            if i > 0:
                self.layers[str(resolution)] = ConvConvDownBlock(
                    in_channels,
                    out_channels,
                    fir=fir,
                    down=2,
                    **kwargs,
                )
            else:
                if self.mbstd_group > 1:
                    in_channels += 1
                    mbstd = layers.MinibatchStd(size=self.mbstd_group)
                block = layers.Conv2dBlock(
                    in_channels, out_channels, 3, 1, 1, **kwargs
                )
                if self.mbstd_group > 1:
                    self.layers[str(resolution)] = nn.Sequential(mbstd, block)
                else:
                    self.layers[str(resolution)] = block

                in_features = out_channels * resolution**2
                out_features = _calc_channels(
                    resolution // 4, self.max_channels, self.channels_decay
                )
                self.layers["last"] = nn.Sequential(
                    layers.LinearBlock(in_features, out_features, **kwargs),
                    layers.Linear(
                        in_features=out_features,
                        out_features=max(self.label_dim, 1),
                        bias=kwargs.get("bias", True),
                    ),
                )

        if lr_equalization:
            hooks.lr_equalizer(module=self, lr_multiplier=lr_multiplier)

    def forward(
        self,
        input: Tensor,
        label: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            input: (batch_size, image_channels, resolution, resolution)
        """
        resolution = input.size(-1)

        output: Tensor = self.from_images[str(resolution)](input)
        for i in range(int(math.log2(resolution)) - 2, -1, -1):
            current_resolution = self.initial_resolution * 2**i
            output = self.layers[str(current_resolution)](output)
        output = self.layers["last"](output)
        if label is not None:
            output = (output * label).sum(dim=1, keepdim=True)
        return output


# ------------------------------------------------------------------------------


def _calc_channels(resolution: int, max: int = 512, decay: float = 1.0) -> int:
    return min(
        max,
        int(2 ** (14 - math.log2(resolution) * decay)),
    )


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
            style_dim: int = 512,
            label_dim: int = 0,
            resolution: int = 1024,
        ) -> None:
            super().__init__()
            self.generator = Generator(
                latent_dim=latent_dim,
                style_dim=style_dim,
                label_dim=label_dim,
                resolution=resolution,
            )
            self.discriminator = Discriminator(
                label_dim=label_dim,
                resolution=resolution,
                mbstd_group=4,
            )
            self.example_input_array = (torch.empty(2, latent_dim),)
            if label_dim:
                self.example_input_array += (torch.empty(2, label_dim),)

        def forward(self, input: Tensor, label: Optional[Tensor] = None) -> Tensor:  # type: ignore
            image = self.generator(input, label)
            for _ in range(self.generator.synthesis.n_layers):
                score = self.discriminator(image, label=label)
                image = F.avg_pool2d(image, 2)
            return score

    summary = Summary()
    print(summary)
    print(summarize(summary, max_depth=args["depth"]))


if __name__ == "__main__":
    main()
