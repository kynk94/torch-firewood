"""
StyleGAN

A Style-Based Generator Architecture for Generative Adversarial Networks
https://arxiv.org/abs/1812.04948

Differences from the official implementation:
    official:
        Uses weight size of noise as 'inputs channels'.
        By the way, official StyleGAN2 uses weight size of noise as '1'.
    this:
        Uses weight size of noise as '1' for convenience, following to the
        implementation of official StyleGAN2.

Therefore, the difference in the total number of parameters come from
the noise layers.
See bottom of this file to check the number of parameters of official StyleGAN.
"""
import argparse
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from firewood import hooks, layers, utils
from firewood.common.types import INT, NUMBER


class MappingNetwork(nn.Module):
    """
    Mapping Network of StyleGAN

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


class InitialBlock(nn.Module):
    def __init__(
        self,
        style_dim: int,
        in_channels: int,
        initial_resolution: int = 4,
        bias: bool = True,
        activation: str = "lrelu",
        noise: Optional[str] = "normal",
        trainable_input: bool = False,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.in_channels = in_channels
        self.initial_resolution = initial_resolution
        self.input = nn.Parameter(
            torch.randn(
                self.in_channels,
                self.initial_resolution,
                self.initial_resolution,
            ),
            requires_grad=trainable_input,
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.in_channels))
        else:
            self.register_buffer("bias", None)
        self.activation = layers.activations.get(activation)
        self.noise = layers.noise.get(noise)
        self.adain = layers.AdaptiveNorm(
            self.in_channels, self.style_dim, use_projection=True
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        input: style vector (w)
        """
        output = self.input.expand(input.size(0), -1, -1, -1)
        if self.noise is not None:
            output = self.noise(output)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        if self.activation is not None:
            output = self.activation(output)
        output = self.adain(output, input)
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
        fir: NUMBER = [1, 2, 1],
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
            op_order="WAN",
        )
        self.conv_adain_1 = layers.Conv2dBlock(
            in_channels, out_channels, fir=fir, fir_args=dict(up=up), **kwargs
        )
        self.conv_adain_1.update_layer_in_order(
            "normalization",
            layers.AdaptiveNorm(out_channels, style_dim, use_projection=True),
        )
        self.conv_adain_2 = layers.Conv2dBlock(
            out_channels, out_channels, **kwargs
        )
        self.conv_adain_2.update_layer_in_order(
            "normalization",
            layers.AdaptiveNorm(out_channels, style_dim, use_projection=True),
        )

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        output = self.conv_adain_1(input, style)
        output = self.conv_adain_2(output, style)
        return output


class SynthesisNetwork(nn.Module):
    initial_resolution = 4

    def __init__(
        self,
        style_dim: int = 512,
        image_channels: int = 3,
        max_channels: int = 512,
        channels_decay: float = 1.0,
        resolution: INT = 1024,
        initial_input_type: str = "constant",
        activation: str = "lrelu",
        noise: Optional[str] = "normal",
        fir: NUMBER = [1, 2, 1],
        lr_equalization: bool = True,
        lr_multiplier: float = 0.01,
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.image_channels = image_channels
        self.max_channels = max_channels
        self.channels_decay = channels_decay
        self.resolution = self._check_resolution(resolution)
        self.initial_input_type = initial_input_type
        self.n_layers = int(math.log2(resolution)) - 1

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
                in_channels = self.get_channels(self.initial_resolution)
                out_channels = in_channels
                self.layers["initial"] = self._get_initial_block(in_channels)
                block = layers.Conv2dBlock(
                    in_channels, out_channels, op_order="WAN", **conv_kwargs
                )
                block.update_layer_in_order(
                    "normalization",
                    layers.AdaptiveNorm(
                        in_channels, style_dim, use_projection=True
                    ),
                )
                self.layers[str(resolution)] = block
            else:
                in_channels = out_channels
                out_channels = self.get_channels(resolution)
                self.layers[str(resolution)] = UpConvConvBlock(
                    style_dim,
                    in_channels,
                    out_channels,
                    fir=fir,
                    up=2,
                    **conv_kwargs,
                )

            self.to_images[str(resolution)] = layers.GFixConv2d(
                out_channels, self.image_channels, 3, 1, "same", bias=True
            )

        if lr_equalization:
            hooks.lr_equalizer(self, lr_multiplier=lr_multiplier)

    def forward(
        self,
        input: Tensor,
        alpha: float = 1.0,
        resolution: Optional[int] = None,
    ) -> Tensor:
        """
        input: style vector (w)
        """
        if resolution is None:
            resolution = self.resolution

        for name, layer in self.layers.items():
            if name == "initial":
                output = layer(input)
                continue

            prev_output = output
            output = layer(output, input)
            if name != str(resolution):
                continue

            image = self.to_images[str(resolution)](output)
            if 0.0 <= alpha < 1.0:
                lower_image = self.to_images[str(resolution // 2)](prev_output)
                upsampled_lower_image = utils.image.upsample(
                    lower_image, factor=2, mode="nearest"
                )
                image = (1.0 - alpha) * upsampled_lower_image + alpha * image
            output = image
            break
        return output

    def generate_all_resolution(self, input: Tensor) -> Tuple[Tensor, ...]:
        images = []
        for name, layer in self.layers.items():
            if name == "initial":
                output = layer(input)
                continue
            output = self.layers[name](output, input)
            images.append(self.to_images[name](output))
        return tuple(images)

    def get_channels(self, resolution: int) -> int:
        return min(
            self.max_channels,
            int(2 ** (14 - math.log2(resolution) * self.channels_decay)),
        )

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

    def _get_initial_block(self, in_channels: int) -> nn.Module:
        kwargs = dict(
            in_channels=in_channels,
            style_dim=self.style_dim,
            initial_resolution=self.initial_resolution,
        )
        if self.initial_input_type == "constant":
            return InitialBlock(trainable_input=False, **kwargs)
        if self.initial_input_type == "trainable":
            return InitialBlock(trainable_input=True, **kwargs)
        raise ValueError(
            "Currently one of {'constant', 'trainable'} is supported as "
            f"`initial_input_type`. Received {self.initial_input_type}"
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
            self.mapping = MappingNetwork(
                latent_dim=latent_dim,
                label_dim=label_dim,
                hidden_dim=latent_dim,
                style_dim=style_dim,
                n_layers=8,
            )
            self.synthesis = SynthesisNetwork(
                style_dim=style_dim,
                resolution=resolution,
            )
            self.example_input_array = (torch.empty(2, latent_dim),)
            if label_dim:
                self.example_input_array += (torch.empty(2, label_dim),)

        def forward(self, input: Tensor, label: Optional[Tensor] = None) -> Tensor:  # type: ignore
            style_vector: Tensor = self.mapping(input, label)
            alpha_test = self.synthesis(style_vector, alpha=0.5)
            images = self.synthesis.generate_all_resolution(style_vector)
            return images

    summary = Summary()
    print(summary)
    print(summarize(summary, max_depth=args["depth"]))


if __name__ == "__main__":
    main()


"""
Tensorflow graph of Official implementation of StyleGAN
--------------------------------------------------------------------------------
G                               Params    OutputShape          WeightShape     
---                             ---       ---                  ---             
latents_in                      -         (?, 512)             -               
labels_in                       -         (?, 0)               -               
lod                             -         ()                   -               
dlatent_avg                     -         (512,)               -               
G_mapping/latents_in            -         (?, 512)             -               
G_mapping/labels_in             -         (?, 0)               -               
G_mapping/PixelNorm             -         (?, 512)             -               
G_mapping/Dense0                262656    (?, 512)             (512, 512)      
G_mapping/Dense1                262656    (?, 512)             (512, 512)      
G_mapping/Dense2                262656    (?, 512)             (512, 512)      
G_mapping/Dense3                262656    (?, 512)             (512, 512)      
G_mapping/Dense4                262656    (?, 512)             (512, 512)      
G_mapping/Dense5                262656    (?, 512)             (512, 512)      
G_mapping/Dense6                262656    (?, 512)             (512, 512)      
G_mapping/Dense7                262656    (?, 512)             (512, 512)      
G_mapping/Broadcast             -         (?, 18, 512)         -               
G_mapping/dlatents_out          -         (?, 18, 512)         -               
Truncation                      -         (?, 18, 512)         -               
G_synthesis/dlatents_in         -         (?, 18, 512)         -               
G_synthesis/4x4/Const           534528    (?, 512, 4, 4)       (512,)          
G_synthesis/4x4/Conv            2885632   (?, 512, 4, 4)       (3, 3, 512, 512)
G_synthesis/ToRGB_lod8          1539      (?, 3, 4, 4)         (1, 1, 512, 3)  
G_synthesis/8x8/Conv0_up        2885632   (?, 512, 8, 8)       (3, 3, 512, 512)
G_synthesis/8x8/Conv1           2885632   (?, 512, 8, 8)       (3, 3, 512, 512)
G_synthesis/ToRGB_lod7          1539      (?, 3, 8, 8)         (1, 1, 512, 3)  
G_synthesis/Upscale2D           -         (?, 3, 8, 8)         -               
G_synthesis/Grow_lod7           -         (?, 3, 8, 8)         -               
G_synthesis/16x16/Conv0_up      2885632   (?, 512, 16, 16)     (3, 3, 512, 512)
G_synthesis/16x16/Conv1         2885632   (?, 512, 16, 16)     (3, 3, 512, 512)
G_synthesis/ToRGB_lod6          1539      (?, 3, 16, 16)       (1, 1, 512, 3)  
G_synthesis/Upscale2D_1         -         (?, 3, 16, 16)       -               
G_synthesis/Grow_lod6           -         (?, 3, 16, 16)       -               
G_synthesis/32x32/Conv0_up      2885632   (?, 512, 32, 32)     (3, 3, 512, 512)
G_synthesis/32x32/Conv1         2885632   (?, 512, 32, 32)     (3, 3, 512, 512)
G_synthesis/ToRGB_lod5          1539      (?, 3, 32, 32)       (1, 1, 512, 3)  
G_synthesis/Upscale2D_2         -         (?, 3, 32, 32)       -               
G_synthesis/Grow_lod5           -         (?, 3, 32, 32)       -               
G_synthesis/64x64/Conv0_up      1442816   (?, 256, 64, 64)     (3, 3, 512, 256)
G_synthesis/64x64/Conv1         852992    (?, 256, 64, 64)     (3, 3, 256, 256)
G_synthesis/ToRGB_lod4          771       (?, 3, 64, 64)       (1, 1, 256, 3)  
G_synthesis/Upscale2D_3         -         (?, 3, 64, 64)       -               
G_synthesis/Grow_lod4           -         (?, 3, 64, 64)       -               
G_synthesis/128x128/Conv0_up    426496    (?, 128, 128, 128)   (3, 3, 256, 128)
G_synthesis/128x128/Conv1       279040    (?, 128, 128, 128)   (3, 3, 128, 128)
G_synthesis/ToRGB_lod3          387       (?, 3, 128, 128)     (1, 1, 128, 3)  
G_synthesis/Upscale2D_4         -         (?, 3, 128, 128)     -               
G_synthesis/Grow_lod3           -         (?, 3, 128, 128)     -               
G_synthesis/256x256/Conv0_up    139520    (?, 64, 256, 256)    (3, 3, 128, 64) 
G_synthesis/256x256/Conv1       102656    (?, 64, 256, 256)    (3, 3, 64, 64)  
G_synthesis/ToRGB_lod2          195       (?, 3, 256, 256)     (1, 1, 64, 3)   
G_synthesis/Upscale2D_5         -         (?, 3, 256, 256)     -               
G_synthesis/Grow_lod2           -         (?, 3, 256, 256)     -               
G_synthesis/512x512/Conv0_up    51328     (?, 32, 512, 512)    (3, 3, 64, 32)  
G_synthesis/512x512/Conv1       42112     (?, 32, 512, 512)    (3, 3, 32, 32)  
G_synthesis/ToRGB_lod1          99        (?, 3, 512, 512)     (1, 1, 32, 3)   
G_synthesis/Upscale2D_6         -         (?, 3, 512, 512)     -               
G_synthesis/Grow_lod1           -         (?, 3, 512, 512)     -               
G_synthesis/1024x1024/Conv0_up  21056     (?, 16, 1024, 1024)  (3, 3, 32, 16)  
G_synthesis/1024x1024/Conv1     18752     (?, 16, 1024, 1024)  (3, 3, 16, 16)  
G_synthesis/ToRGB_lod0          51        (?, 3, 1024, 1024)   (1, 1, 16, 3)   
G_synthesis/Upscale2D_7         -         (?, 3, 1024, 1024)   -               
G_synthesis/Grow_lod0           -         (?, 3, 1024, 1024)   -               
G_synthesis/images_out          -         (?, 3, 1024, 1024)   -               
G_synthesis/lod                 -         ()                   -               
G_synthesis/noise0              -         (1, 1, 4, 4)         -               
G_synthesis/noise1              -         (1, 1, 4, 4)         -               
G_synthesis/noise2              -         (1, 1, 8, 8)         -               
G_synthesis/noise3              -         (1, 1, 8, 8)         -               
G_synthesis/noise4              -         (1, 1, 16, 16)       -               
G_synthesis/noise5              -         (1, 1, 16, 16)       -               
G_synthesis/noise6              -         (1, 1, 32, 32)       -               
G_synthesis/noise7              -         (1, 1, 32, 32)       -               
G_synthesis/noise8              -         (1, 1, 64, 64)       -               
G_synthesis/noise9              -         (1, 1, 64, 64)       -               
G_synthesis/noise10             -         (1, 1, 128, 128)     -               
G_synthesis/noise11             -         (1, 1, 128, 128)     -               
G_synthesis/noise12             -         (1, 1, 256, 256)     -               
G_synthesis/noise13             -         (1, 1, 256, 256)     -               
G_synthesis/noise14             -         (1, 1, 512, 512)     -               
G_synthesis/noise15             -         (1, 1, 512, 512)     -               
G_synthesis/noise16             -         (1, 1, 1024, 1024)   -               
G_synthesis/noise17             -         (1, 1, 1024, 1024)   -               
images_out                      -         (?, 3, 1024, 1024)   -               
---                             ---       ---                  ---             
Total                           26219627
--------------------------------------------------------------------------------
D                     Params    OutputShape          WeightShape     
---                   ---       ---                  ---             
images_in             -         (?, 3, 1024, 1024)   -               
labels_in             -         (?, 0)               -               
lod                   -         ()                   -               
FromRGB_lod0          64        (?, 16, 1024, 1024)  (1, 1, 3, 16)   
1024x1024/Conv0       2320      (?, 16, 1024, 1024)  (3, 3, 16, 16)  
1024x1024/Conv1_down  4640      (?, 32, 512, 512)    (3, 3, 16, 32)  
Downscale2D           -         (?, 3, 512, 512)     -               
FromRGB_lod1          128       (?, 32, 512, 512)    (1, 1, 3, 32)   
Grow_lod0             -         (?, 32, 512, 512)    -               
512x512/Conv0         9248      (?, 32, 512, 512)    (3, 3, 32, 32)  
512x512/Conv1_down    18496     (?, 64, 256, 256)    (3, 3, 32, 64)  
Downscale2D_1         -         (?, 3, 256, 256)     -               
FromRGB_lod2          256       (?, 64, 256, 256)    (1, 1, 3, 64)   
Grow_lod1             -         (?, 64, 256, 256)    -               
256x256/Conv0         36928     (?, 64, 256, 256)    (3, 3, 64, 64)  
256x256/Conv1_down    73856     (?, 128, 128, 128)   (3, 3, 64, 128) 
Downscale2D_2         -         (?, 3, 128, 128)     -               
FromRGB_lod3          512       (?, 128, 128, 128)   (1, 1, 3, 128)  
Grow_lod2             -         (?, 128, 128, 128)   -               
128x128/Conv0         147584    (?, 128, 128, 128)   (3, 3, 128, 128)
128x128/Conv1_down    295168    (?, 256, 64, 64)     (3, 3, 128, 256)
Downscale2D_3         -         (?, 3, 64, 64)       -               
FromRGB_lod4          1024      (?, 256, 64, 64)     (1, 1, 3, 256)  
Grow_lod3             -         (?, 256, 64, 64)     -               
64x64/Conv0           590080    (?, 256, 64, 64)     (3, 3, 256, 256)
64x64/Conv1_down      1180160   (?, 512, 32, 32)     (3, 3, 256, 512)
Downscale2D_4         -         (?, 3, 32, 32)       -               
FromRGB_lod5          2048      (?, 512, 32, 32)     (1, 1, 3, 512)  
Grow_lod4             -         (?, 512, 32, 32)     -               
32x32/Conv0           2359808   (?, 512, 32, 32)     (3, 3, 512, 512)
32x32/Conv1_down      2359808   (?, 512, 16, 16)     (3, 3, 512, 512)
Downscale2D_5         -         (?, 3, 16, 16)       -               
FromRGB_lod6          2048      (?, 512, 16, 16)     (1, 1, 3, 512)  
Grow_lod5             -         (?, 512, 16, 16)     -               
16x16/Conv0           2359808   (?, 512, 16, 16)     (3, 3, 512, 512)
16x16/Conv1_down      2359808   (?, 512, 8, 8)       (3, 3, 512, 512)
Downscale2D_6         -         (?, 3, 8, 8)         -               
FromRGB_lod7          2048      (?, 512, 8, 8)       (1, 1, 3, 512)  
Grow_lod6             -         (?, 512, 8, 8)       -               
8x8/Conv0             2359808   (?, 512, 8, 8)       (3, 3, 512, 512)
8x8/Conv1_down        2359808   (?, 512, 4, 4)       (3, 3, 512, 512)
Downscale2D_7         -         (?, 3, 4, 4)         -               
FromRGB_lod8          2048      (?, 512, 4, 4)       (1, 1, 3, 512)  
Grow_lod7             -         (?, 512, 4, 4)       -               
4x4/MinibatchStddev   -         (?, 513, 4, 4)       -               
4x4/Conv              2364416   (?, 512, 4, 4)       (3, 3, 513, 512)
4x4/Dense0            4194816   (?, 512)             (8192, 512)     
4x4/Dense1            513       (?, 1)               (512, 1)        
scores_out            -         (?, 1)               -               
---                   ---       ---                  ---             
Total                 23087249
--------------------------------------------------------------------------------
"""