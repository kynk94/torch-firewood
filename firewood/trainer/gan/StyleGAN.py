import argparse
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchvision import transforms

from firewood.common.backend import set_runtime_build
from firewood.common.types import INT, NUMBER
from firewood.models.gan.StyleGAN import (
    Discriminator,
    MappingNetwork,
    SynthesisNetwork,
)
from firewood.trainer.callbacks import (
    ExponentialMovingAverage,
    LatentDimInterpolator,
    LatentImageSampler,
    ModelCheckpoint,
)
from firewood.trainer.losses import (
    logistic_nonsaturating_loss,
    simple_gradient_penalty,
)
from firewood.trainer.metrics import FrechetInceptionDistance
from firewood.trainer.utils import find_checkpoint
from firewood.utils.data import (
    NoClassImageFolder,
    get_dataloaders,
    get_train_test_val_datasets,
    torchvision_train_test_val_datasets,
)


class StyleGAN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 512,
        label_dim: int = 0,
        style_dim: int = 512,
        max_channels: int = 512,
        initial_input_type: str = "constant",
        activation: str = "lrelu",
        noise: str = "normal",
        fir: NUMBER = [1, 2, 1],
        mbstd_group: int = 4,
        resolution: INT = 1024,
        image_channels: int = 3,
        lr_equalization: bool = True,
        learning_rate: float = 2e-4,
        r1_gamma: float = 10.0,
        r2_gamma: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mapping = MappingNetwork(
            latent_dim=latent_dim,
            label_dim=label_dim,
            style_dim=style_dim,
            activation=activation,
            lr_equalization=lr_equalization,
            lr_multiplier=0.01,
        )
        self.synthesis = SynthesisNetwork(
            style_dim=style_dim,
            max_channels=max_channels,
            initial_input_type=initial_input_type,
            activation=activation,
            noise=noise,
            fir=fir,
            resolution=resolution,
            image_channels=image_channels,
            lr_equalization=lr_equalization,
            lr_multiplier=1.0,
        )
        self.discriminator = Discriminator(
            label_dim=label_dim,
            max_channels=max_channels,
            mbstd_group=mbstd_group,
            activation=activation,
            fir=fir,
            resolution=resolution,
            image_channels=image_channels,
            lr_equalization=lr_equalization,
            lr_multiplier=1.0,
        )

        # metrics
        self.fid = FrechetInceptionDistance()

    def forward(self, input: Tensor) -> Tensor:
        return self.synthesis(input)

    def generate_latent(self, batch_size: int) -> Tensor:
        return torch.randn(
            size=(batch_size, self.hparams.latent_dim), device=self.device
        )

    def generator_step(self, input: Tensor) -> Dict[str, Tensor]:
        latent = self.generate_latent(input.size(0))
        style: Tensor = self.mapping(latent)
        generated_image: Tensor = self.synthesis(style)
        score_fake: Tensor = self.discriminator(generated_image)
        loss = logistic_nonsaturating_loss(score_fake, True)
        return {"loss/gen": loss}

    def discriminator_step(self, input: Tensor) -> Dict[str, Tensor]:

        latent = self.generate_latent(input.size(0))
        with torch.no_grad():
            generated_image: Tensor = self.synthesis(latent)
        score_fake: Tensor = self.discriminator(generated_image)
        score_real: Tensor = self.discriminator(input)

        loss_fake = logistic_nonsaturating_loss(score_fake, False)
        loss_real = logistic_nonsaturating_loss(score_real, True)
        loss = loss_fake + loss_real

        log_dict = dict()
        if self.hparams.r1_gamma != 0.0:
            r1_penalty = simple_gradient_penalty(score_real, input)
            r1_penalty *= self.hparams.r1_gamma / 2
            loss += r1_penalty
            log_dict.update({"loss/dis_r1_penalty": r1_penalty})
        if self.hparams.r2_gamma != 0.0:
            r2_penalty = simple_gradient_penalty(score_fake, generated_image)
            r2_penalty *= self.hparams.r2_gamma / 2
            loss += r2_penalty
            log_dict.update({"loss/dis_r2_penalty": r2_penalty})

        log_dict.update(
            {
                "loss/dis": loss,
                "score/real": score_real.mean(),
                "score/fake": score_fake.mean(),
            }
        )
        return log_dict

    def training_step(
        self, batch: Tensor, batch_idx: int, optimizer_idx: int
    ) -> Tensor:
        real_images, _ = batch
        if optimizer_idx == 0:
            log_dict = self.discriminator_step(real_images)
            key = "loss/dis"
        else:
            log_dict = self.generator_step(real_images)
            key = "loss/gen"
        loss = log_dict.pop(key)
        self.log(key, loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tensor, batch_idx: int
    ) -> Dict[str, Tensor]:
        real_images, _ = batch

        latent = self.generate_latent(real_images.size(0))
        generated_image = self.synthesis(latent)
        score_fake: Tensor = self.discriminator(generated_image)
        score_real: Tensor = self.discriminator(real_images)

        loss_fake = logistic_nonsaturating_loss(score_fake, False)
        loss_real = logistic_nonsaturating_loss(score_real, True)
        loss = loss_fake + loss_real

        log_dict = dict()
        if self.hparams.r1_gamma != 0.0:
            r1_penalty = simple_gradient_penalty(score_real, input)
            r1_penalty *= self.hparams.r1_gamma / 2
            loss += r1_penalty
            log_dict.update({"val/loss_r1_penalty": r1_penalty})
        if self.hparams.r2_gamma != 0.0:
            r2_penalty = simple_gradient_penalty(score_fake, generated_image)
            r2_penalty *= self.hparams.r2_gamma / 2
            loss += r2_penalty
            log_dict.update({"val/loss_r2_penalty": r2_penalty})

        self.fid.update(real_images, True)
        self.fid.update(generated_image, False)
        log_dict.update(
            {
                "val/loss": loss,
                "val/score_real": score_real.mean(),
                "val/score_fake": score_fake.mean(),
            }
        )
        return log_dict

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        outputs_cache = defaultdict(list)
        for output in outputs:
            for key, value in output.items():
                outputs_cache[key].append(value)
        log_dict = {
            key: torch.stack(value).mean()
            for key, value in outputs_cache.items()
        }
        log_dict["val/fid"] = self.fid.compute()
        self.log_dict(log_dict)

    def configure_optimizers(self) -> Tuple[Any]:
        lr = self.hparams.learning_rate
        generator_parameters = chain(
            self.mapping.parameters(), self.synthesis.parameters()
        )
        generator_optimizer = torch.optim.Adam(
            generator_parameters, lr=lr, betas=(0.0, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.0, 0.999)
        )
        # discriminator first, generator second
        return (
            {"optimizer": discriminator_optimizer},
            {"optimizer": generator_optimizer},
        )


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", type=str,
                             help="Input Datasets Directory")
    input_group.add_argument("--dataset", "-d", type=str,
                             help="Dataset Name predefined in torchvision")
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument("--epoch", "-e", type=int, default=-1)
    step_group.add_argument("--step", "-s", type=int, default=-1)
    parser.add_argument("--checkpoint", "-ckpt", type=str, default=None)
    parser.add_argument("--resolution", "-r", type=int, default=112)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--latent_dim", "-l", type=int, default=1024)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4)
    parser.add_argument("--runtime_build", "-rb", action="store_true")
    args = vars(parser.parse_args())
    # fmt: on

    if args["runtime_build"]:
        set_runtime_build(True)

    transform = []
    if args["resolution"] is not None:
        transform.append(transforms.Resize(args["resolution"]))
    transform.extend([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    transform = transforms.Compose(transform)

    if args["input"]:
        datasets = get_train_test_val_datasets(
            root=args["input"],
            dataset_class=NoClassImageFolder,
            transform=transform,
            loader_mode="RGB",
            split="train/val",
        )
    else:
        datasets = torchvision_train_test_val_datasets(
            name=args["dataset"], root="./datasets", transform=transform
        )
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
        datasets=datasets,
        batch_size=args["batch_size"],
        shuffle=True,
        # when pin_memory=True, data will be pinned to the rank 0 gpu
        pin_memory=False,
    )

    sample_image: Tensor = next(iter(train_dataloader))[0]  # (N, C, H, W)
    channels, resolution = sample_image.shape[1:3]

    if args["checkpoint"] is not None:
        lsgan = StyleGAN.load_from_checkpoint(
            find_checkpoint(args["checkpoint"])
        )
    else:
        lsgan = StyleGAN(
            latent_dim=512,
            label_dim=0,
            style_dim=512,
            max_channels=512,
            initial_input_type="constant",
            activation="lrelu",
            noise="normal",
            fir=[1, 2, 1],
            mbstd_group=4,
            resolution=1024,
            image_channels=3,
            lr_equalization=True,
            learning_rate=2e-4,
        )

    callbacks = [
        ExponentialMovingAverage(),
        ModelCheckpoint(save_last_k=3),
        LatentImageSampler(step=500, add_fixed_samples=True, save_image=True),
        LatentDimInterpolator(
            ndim_to_interpolate=args["latent_dim"] // 10, save_image=True
        ),
    ]
    gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=args["epoch"],
        max_steps=args["step"],
        precision=32,
        check_val_every_n_epoch=5,
        callbacks=callbacks,
        strategy="ddp" if gpus > 1 else None,
    )
    trainer.fit(
        lsgan,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader or test_dataloader,
    )


if __name__ == "__main__":
    main()
