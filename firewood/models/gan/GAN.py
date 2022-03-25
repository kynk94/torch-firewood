import argparse
import math

import torch
import torch.nn as nn
from torch import Tensor

from firewood import layers, utils
from firewood.common.types import INT


class Generator(nn.Module):
    """
    Generator of GAN
    Generative Adversarial Networks https://arxiv.org/abs/1406.2661

    Default values are for MNIST.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        resolution: INT = (28, 28),
        channels: int = 1,
    ) -> None:
        super().__init__()
        image_shape = (channels, *utils._pair(resolution))
        out_dim = math.prod(image_shape)

        self.layers = nn.ModuleList(
            [
                layers.LinearBlock(latent_dim, 128, activation="lrelu"),
                layers.LinearBlock(128, 256, activation="lrelu"),
                layers.LinearBlock(256, 256, activation="lrelu"),
                layers.LinearBlock(256, out_dim, activation="tanh"),
                layers.Reshape(image_shape),
            ]
        )

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


class Discriminator(nn.Module):
    """
    Discriminator of GAN
    Generative Adversarial Networks https://arxiv.org/abs/1406.2661

    Default values are for MNIST.
    """

    def __init__(
        self,
        resolution: INT = (28, 28),
        channels: int = 1,
    ) -> None:
        super().__init__()
        image_shape = (channels, *utils._pair(resolution))
        in_dim = math.prod(image_shape)

        self.layers = nn.ModuleList(
            [
                layers.Flatten(),
                layers.LinearBlock(in_dim, 256, activation="lrelu"),
                layers.LinearBlock(256, 256, activation="lrelu"),
                layers.LinearBlock(256, 128, activation="lrelu"),
                layers.Linear(128, 1),
            ]
        )

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
            latent_dim: int = 32,
            resolution: INT = 28,
            channels: int = 1,
        ) -> None:
            super().__init__()
            self.generator = Generator(latent_dim, resolution, channels)
            self.discriminator = Discriminator(resolution, channels)
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
