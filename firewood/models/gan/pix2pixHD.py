import argparse
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from firewood import layers, utils
from firewood.common.types import INT
from firewood.models.gan.pix2pix import PatchGAN


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        padding_mode="reflect",
        normalization="in",
        activation="relu",
    ) -> None:
        super().__init__()

        # fmt: off
        conv_kwargs = dict(kernel_size=kernel_size, stride=stride, padding="same", bias=bias,
                           padding_mode=padding_mode, normalization=normalization)
        self.layers = nn.ModuleList([
            layers.Conv2dBlock(channels, channels, activation=activation, **conv_kwargs),
            layers.Conv2dBlock(channels, channels, **conv_kwargs),
        ])
        # fmt: on

    def forward(self, input: Tensor) -> Tensor:
        # input -> conv -> norm -> act -> conv -> norm -> output + input
        output = input
        for layer in self.layers:
            output = layer(output)
        return input + output


class GlobalGenerator(nn.Module):
    """
    GlobalGenerator of Pix2PixHD
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    https://arxiv.org/abs/1711.11585
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        n_filters: int = 64,
        n_down_blocks: int = 4,
        n_res_blocks: int = 9,
        padding_mode: str = "reflect",
        normalization: str = "in",
        activation: str = "relu",
        max_filters: int = 2**10,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        n_filters = min(max_filters, n_filters)

        # fmt: off
        kwargs = dict(bias=True, padding_mode=padding_mode, normalization=normalization, activation=activation)
        # Downsample Blocks
        self.downsamples = nn.ModuleList([
            layers.Conv2dBlock(in_channels, n_filters, 7, 1, 0,
                               normalization=normalization, activation=activation)
        ])
        for i in range(n_down_blocks):
            in_channels = min(max_filters, n_filters * 2**i)
            out_channels = min(max_filters, in_channels * 2)
            self.downsamples.append(
                layers.Conv2dBlock(in_channels, out_channels, 3, 2, 1, **kwargs)
            )

        # Residual Blocks
        self.residuals = nn.ModuleList()
        res_channels = min(max_filters, n_filters * 2**n_down_blocks)
        for i in range(n_res_blocks):
            self.residuals.append(ResBlock(res_channels, 3, 1, **kwargs))

        # Upsample Blocks
        self.upsamples = nn.ModuleList()
        for i in range(n_down_blocks):
            in_channels = n_filters * 2**(n_down_blocks - i)
            out_channels = in_channels // 2
            if in_channels > max_filters:
                in_channels = max_filters
                out_channels = max_filters
            self.upsamples.append(
                layers.ConvTranspose2dBlock(in_channels, out_channels, 3, 2, 1, output_padding=1,
                                            bias=True, normalization=normalization, activation=activation)
            )
        self.upsamples.append(layers.Conv2dBlock(n_filters, self.out_channels, 7, 1, "same", bias=True,
                                                padding_mode=padding_mode, activation="tanh"))
        # fmt: on

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for downsample in self.downsamples:
            output = downsample(output)
        for residual in self.residuals:
            output = residual(output)
        for upsample in self.upsamples:
            output = upsample(output)
        return output


class LocalEnhancer(nn.Module):
    """
    LocalEnhancer of Pix2PixHD
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    https://arxiv.org/abs/1711.11585

    Operation Graph:
        AvgPool(input)**n -> GlobalGenerator = output_0\n
        AvgPool(input)**(n-1) -> Downsample -> Add(output_0) -> Upsample -> output_1\n
        ...\n
        AvgPool(input)**1 -> Downsample -> Add(output_(n-2)) -> Upsample -> output_(n-1)\n
        input             -> Downsample -> Add(output_(n-1)) -> Upsample -> output_n\n
        OutputLayer(output_n) -> output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        n_local_filters: int = 32,
        n_local_enhancers: int = 1,
        n_down_blocks: int = 4,
        n_local_res_blocks: int = 3,
        n_global_res_blocks: int = 9,
        padding_mode: str = "reflect",
        normalization: str = "in",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.n_local_filters = n_local_filters
        self.n_local_enhancers = n_local_enhancers

        self.global_generator = GlobalGenerator(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            n_filters=self.n_local_filters * 2**self.n_local_enhancers,
            n_down_blocks=n_down_blocks,
            n_res_blocks=n_global_res_blocks,
            padding_mode=padding_mode,
            normalization=normalization,
            activation=activation,
            max_filters=2**10,
        )
        # Remove last convolution layer of GlobalGenerator to extract features
        self.global_generator.upsamples = self.global_generator.upsamples[:-1]
        for param in self.global_generator.parameters():
            param.requires_grad = False

        # fmt: off
        self.downsamples = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        downsample_kwargs = dict(bias=True, padding_mode=padding_mode, normalization=normalization, activation=activation)
        residual_kwargs = dict(bias=True, padding_mode=padding_mode, normalization=normalization, activation=activation)
        upsample_kwargs = dict(bias=True, normalization=normalization, activation=activation)
        for i in range(self.n_local_enhancers):
            global_filters = self.n_local_filters * 2**i

            self.downsamples.append(nn.ModuleList([
                layers.Conv2dBlock(self.in_channels, global_filters, 7, 1, 3, **downsample_kwargs),
                layers.Conv2dBlock(global_filters, 2 * global_filters, 3, 2, 1, **downsample_kwargs)
            ]))

            self.residuals.append(nn.ModuleList([
                ResBlock(2 * global_filters, 3, 1, **residual_kwargs)
                for _ in range(n_local_res_blocks)
            ]))

            self.upsamples.append(nn.ModuleList([
                layers.ConvTranspose2dBlock(2 * global_filters, global_filters, 3, 2, 1, output_padding=1, **upsample_kwargs),
            ]))

        self.output_layer = layers.Conv2dBlock(self.n_local_filters, self.out_channels, 7, 1, "same", bias=True,
                                               padding_mode=padding_mode, activation="tanh")
        # fmt: on

    def forward(self, input: Tensor) -> Tensor:
        inputs = [input]
        for _ in range(self.n_local_enhancers):
            downsampled: Tensor = F.avg_pool2d(
                inputs[-1], 3, stride=2, padding=1, count_include_pad=False
            )
            inputs.append(downsampled)

        # GlobalGenerator features
        self.global_generator.eval()
        output: Tensor = self.global_generator(inputs[-1])

        for i in range(self.n_local_enhancers - 1, -1, -1):
            downsampled = inputs[i]
            for downsample in self.downsamples[i]:
                downsampled = downsample(downsampled)

            # Add GlobalGenerator features to local enhancer
            output += downsampled

            for residual in self.residuals[i]:
                output = residual(output)
            for upsample in self.upsamples[i]:
                output = upsample(output)
        return self.output_layer(output)


class Encoder(nn.Module):
    """
    Encoder of Pix2PixHD for instance-wise embedding
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    https://arxiv.org/abs/1711.11585
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        n_filters: int = 16,
        n_down_blocks: int = 4,
        padding_mode: str = "reflect",
        normalization: str = "in",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # fmt: off
        kwargs = dict(bias=True, padding_mode=padding_mode, normalization=normalization, activation=activation)
        self.layers = nn.ModuleList(
            [layers.Conv2dBlock(self.in_channels, n_filters, 7, 1, 3, **kwargs)]
        )
        for i in range(n_down_blocks):
            in_channels = n_filters * 2**i
            out_channels = in_channels * 2
            self.layers.append(
                layers.Conv2dBlock(in_channels, out_channels, 3, 2, 1, **kwargs)
            )

        for i in range(n_down_blocks, 0, -1):
            in_channels = n_filters * 2**i
            out_channels = in_channels // 2
            self.layers.append(
                layers.ConvTranspose2dBlock(in_channels, out_channels, 3, 2, 1, output_padding=1,
                                            bias=True, normalization=normalization, activation=activation)
            )
        self.layers.append(
            layers.Conv2dBlock(n_filters, self.out_channels, 7, 1, 3, 
                               **utils.updated_dict(kwargs, activation="tanh"))
        )
        # fmt: on

    def forward(
        self, input: Tensor, instance_input: Tensor, normalize: bool = False
    ) -> Tensor:
        """
        input: (N, C, H, W), range [-1, 1]
        instance_input: (N, 1, H, W), range [0, 255]
        normalize: normalize instance_input to [0, 255] if range in [-1, 1]
        """
        output = input
        for layer in self.layers:
            output: Tensor = layer(output)

        # instance-wise average pooling
        mean_output = output.clone()
        if normalize:
            instance_input = instance_input.mul(127.5).add(128).clamp_(0, 255)
        instance_input = instance_input.byte()
        instance_list = instance_input.unique()
        for instance in instance_list:
            for b in range(output.size(0)):
                indices = (instance_input[b : b + 1] == instance).nonzero()
                for j in range(self.out_channels):
                    output_instance = output[
                        indices[:, 0] + b,
                        indices[:, 1] + j,
                        indices[:, 2],
                        indices[:, 3],
                    ]
                    mean_feature = torch.mean(output_instance).expand_as(
                        output_instance
                    )
                    mean_output[
                        indices[:, 0] + b,
                        indices[:, 1] + j,
                        indices[:, 2],
                        indices[:, 3],
                    ] = mean_feature
        return mean_output


class MultiScalePatchGAN(nn.Module):
    """
    Multi-Scale Discriminator of Pix2PixHD
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    https://arxiv.org/abs/1711.11585
    """

    def __init__(
        self,
        in_channels: int = 6,
        n_layers: int = 4,
        n_discriminators: int = 3,
        normalization: str = "in",
        activation: str = "lrelu",
    ) -> None:
        super().__init__()
        self.n_discriminators = n_discriminators

        # fmt: off
        self.discriminators = nn.ModuleList()
        for _ in range(self.n_discriminators):
            self.discriminators.append(
                PatchGAN(in_channels, n_layers=n_layers, normalization=normalization, activation=activation)
            )
        # fmt: on

    def avg_pool(self, *inputs: Tensor) -> Tuple[Tensor, ...]:
        outputs: List[Tensor] = []
        for input in inputs:
            outputs.append(
                F.avg_pool2d(input, 3, 2, 1, count_include_pad=False)
            )
        return tuple(outputs)

    def forward(
        self, input: Tensor, label: Tensor, extract_features: bool = False
    ) -> Union[List[Tensor], List[List[Tensor]]]:
        inputs = [(input, label)]
        for _ in range(self.n_discriminators - 1):
            inputs.append(self.avg_pool(*inputs[-1]))

        outputs: Union[List[Tensor], List[List[Tensor]]] = []
        for i in range(self.n_discriminators):
            outputs.append(self.discriminators[i](*inputs[i], extract_features))
        return outputs


def main() -> None:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.model_summary import summarize

    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", "-d", type=int, default=3)
    args = vars(parser.parse_args())

    class Summary(pl.LightningModule):
        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 3,
            instance_out_channels: int = 3,
            resolution: INT = (1024, 512),
        ) -> None:
            super().__init__()
            self.encoder = Encoder(out_channels, instance_out_channels)
            self.local_enhancer = LocalEnhancer(
                in_channels + instance_out_channels,
                out_channels,
                n_local_enhancers=2,
            )
            self.discriminator = MultiScalePatchGAN(
                in_channels + out_channels, n_discriminators=2
            )
            self.example_input_array = (
                torch.empty(2, in_channels, *utils._pair(resolution)),
                torch.empty(2, out_channels, *utils._pair(resolution)),
                True,
            )

        def forward(
            self, source: Tensor, target: Tensor, extract_features: bool = False
        ) -> List[Tensor]:  # type: ignore
            feature_map = self.encoder(target, target[:, :1], True)
            input = torch.cat((source, feature_map), dim=1)
            generated_image: Tensor = self.local_enhancer(input)
            score: List[Tensor] = self.discriminator(
                source, generated_image, extract_features
            )
            return score

    summary = Summary()
    print(summary)
    print(summarize(summary, max_depth=args["depth"]))


if __name__ == "__main__":
    main()
