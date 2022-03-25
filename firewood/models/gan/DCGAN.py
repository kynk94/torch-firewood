import argparse
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from firewood import layers, utils
from firewood.common.types import INT


class Generator(nn.Module):
    """
    Generator of DCGAN
    Unsupervised Representation Learning with Deep Convolutional
    Generative Adversarial Networks https://arxiv.org/abs/1511.06434
    """

    def __init__(
        self,
        latent_dim: int = 100,
        n_layers: int = 4,
        n_filters: int = 1024,
        normalization: str = "bn",
        activation: str = "relu",
        fir: Optional[List[float]] = None,
        resolution: INT = (64, 64),
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.normalization = normalization
        self.resolution = utils._pair(resolution)
        init_resolution = self.resolution[0] // 2**n_layers

        self.layers = nn.ModuleList()
        # fmt: off
        self.layers.extend([
            layers.LinearBlock(self.latent_dim, n_filters*init_resolution**2,
                               activation=activation),
            layers.Reshape(n_filters, init_resolution, init_resolution),
        ])
        # fmt: on
        for i in range(n_layers):
            if i == 0:
                in_channels = n_filters
                out_channels = n_filters // 2
                normalization = self.normalization
            elif i < n_layers - 1:
                in_channels = out_channels
                out_channels = max(out_channels // 2, channels)
            else:
                in_channels = out_channels
                out_channels = channels
                normalization = None
                activation = "tanh"
            # fmt: off
            self.layers.append(
                layers.ConvTranspose2dBlock(
                    in_channels, out_channels, 4, 2, 1,
                    normalization=normalization, activation=activation, fir=fir
                )
            )
            # fmt: on

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


class Discriminator(nn.Module):
    """
    Discriminator of DCGAN
    Unsupervised Representation Learning with Deep Convolutional
    Generative Adversarial Networks https://arxiv.org/abs/1511.06434
    """

    def __init__(
        self,
        n_layers: int = 4,
        n_filters: int = 64,
        normalization: str = "bn",
        activation: str = "lrelu",
        resolution: INT = (64, 64),
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.resolution = utils._pair(resolution)

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_channels = channels
                out_channels = n_filters
            else:
                in_channels = out_channels
                out_channels = min(2 * out_channels, 1024)
            # fmt: off
            self.layers.append(
                layers.Conv2dBlock(
                    in_channels, out_channels, 4, 2, 1,
                    normalization=normalization, activation=activation,
                )
            )
            # fmt: on
        # `kernel_size` makes the spatial shape to 1
        kernel_size = max(self.resolution[0] // 2**n_layers, 1)
        # fmt: off
        self.layers.extend([
            layers.Conv2d(out_channels, 1, kernel_size),
            layers.Flatten(),
        ])
        # fmt: on

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


def main() -> None:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.model_summary import summarize

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", "-d", type=int, default=3)
    args = vars(parser.parse_args())

    class Summary(pl.LightningModule):
        def __init__(
            self,
            latent_dim: int = 100,
            resolution: INT = 64,
            channels: int = 3,
        ) -> None:
            super().__init__()
            self.generator = Generator(
                latent_dim=latent_dim,
                n_layers=4,
                n_filters=1024,
                normalization="bn",
                activation="relu",
                fir=[1, 3, 3, 1],
                resolution=resolution,
                channels=channels,
            )
            self.discriminator = Discriminator(
                n_layers=4,
                n_filters=64,
                normalization="bn",
                activation="lrelu",
                resolution=resolution,
                channels=channels,
            )
            self.example_input_array = torch.empty(2, latent_dim)

        def forward(self, input: Tensor) -> Tensor:  # type: ignore
            generated_image: Tensor = self.generator(input)
            score: Tensor = self.discriminator(generated_image)
            return score

    summary = Summary()
    print(summary)
    print(summarize(summary, max_depth=args["depth"]))


if __name__ == "__main__":
    main()
