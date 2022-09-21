import argparse
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from firewood import hooks, layers, utils
from firewood.common.types import INT, NUMBER


class MappingNetwork(nn.Module):
    """
    Mapping Network of StyleGAN

    A Style-Based Generator Architecture for Generative Adversarial Networks
    https://arxiv.org/abs/1812.04948

    latent_dim (latent z)
        |
    hidden_dim * n_layers (mlp)
        |
    style_dim (style w)
    """

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
        lr_equalization: bool = True,
        lr_multiplier: float = 0.01,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim

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

        if lr_equalization:
            hooks.lr_equalizer(self, lr_multiplier=lr_multiplier)

    def forward(self, input: Tensor, label: Optional[Tensor] = None) -> Tensor:
        if self.label_affine is not None:
            label = self.label_affine(label)
            input = torch.cat([input, label], dim=1)

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
            latent_dim: int = 512,
            style_dim: int = 512,
            label_dim: int = 128,
        ) -> None:
            super().__init__()
            self.mapping = MappingNetwork(
                latent_dim=latent_dim,
                label_dim=label_dim,
                hidden_dim=latent_dim,
                style_dim=style_dim,
                n_layers=8,
            )
            self.example_input_array = (torch.empty(2, latent_dim),)
            if label_dim:
                self.example_input_array += (torch.empty(2, label_dim),)

        def forward(self, input: Tensor, label: Optional[Tensor] = None) -> Tensor:  # type: ignore
            style_vector: Tensor = self.mapping(input, label)
            return style_vector

    summary = Summary()
    print(summary)
    print(summarize(summary, max_depth=args["depth"]))


if __name__ == "__main__":
    main()
