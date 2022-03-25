import argparse
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from firewood import layers, utils
from firewood.common.types import INT


class Generator(nn.Module):
    """
    Generator of LSGAN
    Least Squares Generative Adversarial Networks
    https://arxiv.org/abs/1611.04076
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        n_layer: int = 7,
        n_filter: int = 256,
        normalization: str = "bn",
        activation: str = "relu",
        fir: Optional[List[float]] = None,
        resolution: INT = (112, 112),
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.resolution = utils._pair(resolution)
        init_resolution = self.resolution[0] // 2 ** (n_layer - 3)

        self.layers = nn.ModuleList()
        # fmt: off
        kwargs = dict(bias=True, normalization=normalization, activation=activation, fir=fir)
        self.layers.extend([
            layers.LinearBlock(self.latent_dim, n_filter*init_resolution**2,
                               **utils.updated_dict(kwargs, delete="fir")),
            layers.Reshape(n_filter, init_resolution, init_resolution),
        ])
        for _ in range(2):
            self.layers.extend([
                layers.ConvTranspose2dBlock(n_filter, n_filter, 3, 2, "same", **kwargs),
                layers.ConvTranspose2dBlock(n_filter, n_filter, 3, 1, 1, **kwargs),
            ])
        # fmt: on
        for i in range(n_layer - 5):
            if i == 0:
                in_channels = n_filter
                out_channels = n_filter // 2
            else:
                in_channels = out_channels
                out_channels = max(out_channels // 2, channels)
            # fmt: off
            self.layers.append(
                layers.ConvTranspose2dBlock(in_channels, out_channels, 3, 2, "same", **kwargs)
            )
            # fmt: on
        self.layers.append(
            layers.ConvTranspose2dBlock(
                out_channels, channels, 3, 1, 1, activation="tanh", fir=fir
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


class Discriminator(nn.Module):
    """
    Discriminator of LSGAN
    Least Squares Generative Adversarial Networks
    https://arxiv.org/abs/1611.04076
    """

    def __init__(
        self,
        n_layer: int = 4,
        n_filter: int = 64,
        activation: str = "lrelu",
        resolution: INT = (112, 112),
        channels: int = 3,
    ) -> None:
        super().__init__()
        self.resolution = utils._pair(resolution)

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                in_channels = channels
                out_channels = n_filter
                normalization = None
            else:
                in_channels = out_channels
                out_channels = min(2 * out_channels, 1024)
                normalization = "bn"
            # fmt: off
            self.layers.append(
                layers.Conv2dBlock(
                    in_channels, out_channels, 5, 2, 2,
                    normalization=normalization, activation=activation,
                )
            )
            # fmt: on
        # `kernel_size` makes the spatial shape to 1
        kernel_size = max(self.resolution[0] // 2**n_layer, 1)
        # fmt: off
        self.layers.extend([
            # Bias is meaningless when use mse loss without activation.
            # If activation is need, use `bias=True`
            layers.Conv2d(out_channels, 1, kernel_size, bias=False),
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
            latent_dim: int = 1024,
            resolution: INT = 112,
            channels: int = 3,
        ) -> None:
            super().__init__()
            self.generator = Generator(
                latent_dim=latent_dim,
                n_layer=7,
                n_filter=256,
                normalization="bn",
                activation="relu",
                fir=[1, 2, 1],
                resolution=resolution,
                channels=channels,
            )
            self.discriminator = Discriminator(
                n_layer=4,
                n_filter=64,
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
