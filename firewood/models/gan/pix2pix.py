import argparse
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from firewood import layers, utils
from firewood.common.types import INT


class Generator(nn.Module):
    """
    Generator of Pix2Pix
    Image-to-Image Translation with Conditional Adversarial Networks
    https://arxiv.org/abs/1611.07004
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        # fmt: off
        encoder_kwargs = dict(kernel_size=4, stride=2, padding=1, bias=True,
                              padding_mode="reflect", normalization="bn", activation="lrelu")
        self.encoder = nn.ModuleList([
            layers.Conv2dBlock(in_channels, 64, **utils.updated_dict(encoder_kwargs, normalization=None)),
            layers.Conv2dBlock(64, 128, **encoder_kwargs),
            layers.Conv2dBlock(128, 256, **encoder_kwargs),
            layers.Conv2dBlock(256, 512, **encoder_kwargs),
            layers.Conv2dBlock(512, 512, **encoder_kwargs),
            layers.Conv2dBlock(512, 512, **encoder_kwargs),
            layers.Conv2dBlock(512, 512, **encoder_kwargs),
            layers.Conv2dBlock(512, 512, **utils.updated_dict(encoder_kwargs, normalization=None)),
        ])

        decoder_kwargs = dict(kernel_size=4, stride=2, padding=1, bias=True,
                              normalization="bn", activation="relu")
        self.decoder = nn.ModuleList([
            layers.ConvTranspose2dBlock(512, 512, dropout=0.5, **decoder_kwargs),
            layers.ConvTranspose2dBlock(1024, 512, dropout=0.5, **decoder_kwargs),
            layers.ConvTranspose2dBlock(1024, 512, dropout=0.5, **decoder_kwargs),
            layers.ConvTranspose2dBlock(1024, 512, **decoder_kwargs),
            layers.ConvTranspose2dBlock(1024, 256, **decoder_kwargs),
            layers.ConvTranspose2dBlock(512, 128, **decoder_kwargs),
            layers.ConvTranspose2dBlock(256, 64, **decoder_kwargs),
            layers.ConvTranspose2dBlock(128, out_channels, **utils.updated_dict(decoder_kwargs, normalization=None, activation="tanh")),
        ])
        # fmt: on

    def forward(self, input: Tensor) -> Tensor:
        output = input

        # Encoder
        skip_connections = []
        for layer in self.encoder:
            output = layer(output)
            skip_connections.append(output)
        skip_connections.pop()

        # Decoder
        for layer in self.decoder:
            output = layer(output)
            if skip_connections:
                output = torch.cat([output, skip_connections.pop()], dim=1)
        return output


class PatchGAN(nn.Module):
    """
    PatchGAN of Pix2Pix
    Image-to-Image Translation with Conditional Adversarial Networks
    https://arxiv.org/abs/1611.07004
    """

    def __init__(
        self,
        in_channels: int = 6,
        kernel_size: int = 4,
        n_filters: int = 64,
        n_layers: int = 6,
        padding_mode: str = "reflect",
        normalization: str = "bn",
        activation: str = "lrelu",
        max_filters: int = 2**9,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.max_filters = max_filters
        self.n_filters = min(self.max_filters, n_filters)

        # fmt: off
        kwargs = dict(kernel_size=kernel_size, stride=2, padding="same", bias=True,
                      padding_mode=padding_mode, normalization=normalization, activation=activation)
        self.layers = nn.ModuleList([
            layers.Conv2dBlock(in_channels, self.n_filters, **utils.updated_dict(kwargs, normalization=None)),
        ])
        for i in range(n_layers-1):
            in_channels = min(self.max_filters, self.n_filters * 2**i)
            out_channels = min(self.max_filters, in_channels * 2)
            self.layers.append(layers.Conv2dBlock(in_channels, out_channels, **kwargs))
        # fmt: on
        head_in_channels = min(
            self.max_filters, self.n_filters * 2 ** (n_layers - 1)
        )
        self.head_in_channels = head_in_channels
        self.layers.append(layers.Conv2d(self.head_in_channels, 1, 1))

    def forward(
        self,
        input: Tensor,
        label: Optional[Tensor] = None,
        extract_features: bool = False,
    ) -> Union[Tensor, List[Tensor]]:
        if label is None:
            output = input
        else:
            output = torch.cat([input, label], dim=1)
        if not extract_features:
            for layer in self.layers:
                output = layer(output)
            return output

        features = []
        for layer in self.layers:
            output = layer(output)
            features.append(output)
        return features


class PatchGANx1(PatchGAN):
    """
    1x1 PatchGAN of Pix2Pix
    Image-to-Image Translation with Conditional Adversarial Networks
    https://arxiv.org/abs/1611.07004
    """

    def __init__(
        self,
        in_channels: int = 6,
        n_filters: int = 64,
        padding_mode: str = "reflect",
        normalization: str = "bn",
        activation: str = "lrelu",
        max_filters: int = 2**9,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            kernel_size=1,
            n_filters=n_filters,
            n_layers=2,
            padding_mode=padding_mode,
            normalization=normalization,
            activation=activation,
            max_filters=max_filters,
        )


class PatchGANx16(PatchGAN):
    """
    16x16 PatchGAN of Pix2Pix
    Image-to-Image Translation with Conditional Adversarial Networks
    https://arxiv.org/abs/1611.07004
    """

    def __init__(
        self,
        in_channels: int = 6,
        n_filters: int = 64,
        padding_mode: str = "reflect",
        normalization: str = "bn",
        activation: str = "lrelu",
        max_filters: int = 2**9,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            kernel_size=4,
            n_filters=n_filters,
            n_layers=2,
            padding_mode=padding_mode,
            normalization=normalization,
            activation=activation,
            max_filters=max_filters,
        )


class PatchGANx70(PatchGAN):
    """
    70x70 PatchGAN of Pix2Pix
    Image-to-Image Translation with Conditional Adversarial Networks
    https://arxiv.org/abs/1611.07004
    """

    def __init__(
        self,
        in_channels: int = 6,
        n_filters: int = 64,
        padding_mode: str = "reflect",
        normalization: str = "bn",
        activation: str = "lrelu",
        max_filters: int = 2**9,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            kernel_size=4,
            n_filters=n_filters,
            n_layers=4,
            padding_mode=padding_mode,
            normalization=normalization,
            activation=activation,
            max_filters=max_filters,
        )


class PatchGANx286(PatchGAN):
    """
    286x286 PatchGAN of Pix2Pix
    Image-to-Image Translation with Conditional Adversarial Networks
    https://arxiv.org/abs/1611.07004
    """

    def __init__(
        self,
        in_channels: int = 6,
        n_filters: int = 64,
        padding_mode: str = "reflect",
        normalization: str = "bn",
        activation: str = "lrelu",
        max_filters: int = 2**9,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            kernel_size=4,
            n_filters=n_filters,
            n_layers=6,
            padding_mode=padding_mode,
            normalization=normalization,
            activation=activation,
            max_filters=max_filters,
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
            in_channels: int = 3,
            out_channels: int = 1,
            resolution: INT = 256,
        ) -> None:
            super().__init__()
            self.generator = Generator(in_channels, out_channels)
            self.discriminator_1 = PatchGANx1(in_channels + out_channels)
            self.discriminator_16 = PatchGANx16(in_channels + out_channels)
            self.discriminator_70 = PatchGANx70(in_channels + out_channels)
            self.discriminator_286 = PatchGANx286(in_channels + out_channels)
            self.example_input_array = (
                torch.empty(2, in_channels, *utils._pair(resolution)),
                torch.empty(2, out_channels, *utils._pair(resolution)),
            )

        def forward(self, input: Tensor, label: Tensor) -> Tensor:  # type: ignore
            generated_image: Tensor = self.generator(input)
            concatenated_input = torch.cat((input, generated_image), dim=1)
            scores: List[Tensor] = []
            scores.append(self.discriminator_1(input, generated_image))
            scores.append(self.discriminator_16(input, generated_image))
            scores.append(self.discriminator_70(input, generated_image))
            scores.append(self.discriminator_286(concatenated_input))
            return scores[-1]

    summary = Summary()
    print(summary)
    print(summarize(summary, max_depth=args["depth"]))


if __name__ == "__main__":
    main()
