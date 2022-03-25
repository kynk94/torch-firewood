import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchvision import transforms

from firewood.common.types import INT
from firewood.models.gan.GAN import Discriminator, Generator
from firewood.trainer.callbacks import (
    LatentDimInterpolator,
    LatentImageSampler,
    ModelCheckpoint,
)
from firewood.trainer.losses import gan_loss
from firewood.trainer.metrics import FrechetInceptionDistance
from firewood.utils.data import (
    get_dataloaders,
    get_train_test_val_datasets,
    torchvision_train_test_val_datasets,
)


class GAN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 32,
        resolution: INT = 28,
        channels: int = 1,
        learning_rate: float = 2e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(
            latent_dim=latent_dim,
            resolution=resolution,
            channels=channels,
        )
        self.discriminator = Discriminator(
            resolution=resolution, channels=channels
        )

        # metrics
        self.fid = FrechetInceptionDistance()

    def forward(self, input: Tensor) -> Tensor:
        return self.generator(input)

    def generate_latent(self, batch_size: int) -> Tensor:
        return torch.randn(
            size=(batch_size, self.hparams.latent_dim), device=self.device
        )

    def generator_step(self, input: Tensor) -> Dict[str, Tensor]:
        latent = self.generate_latent(input.size(0))
        generated_image: Tensor = self.generator(latent)
        score_fake: Tensor = self.discriminator(generated_image)
        loss = gan_loss(score_fake, True)
        return {"loss/gen": loss}

    def discriminator_step(self, input: Tensor) -> Dict[str, Tensor]:
        latent = self.generate_latent(input.size(0))
        with torch.no_grad():
            generated_image: Tensor = self.generator(latent)
        score_fake: Tensor = self.discriminator(generated_image)
        score_real: Tensor = self.discriminator(input)

        loss_fake = gan_loss(score_fake, False)
        loss_real = gan_loss(score_real, True)

        loss = loss_fake + loss_real
        return {
            "loss/dis": loss,
            "score/real": score_real.mean(),
            "score/fake": score_fake.mean(),
        }

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
        generated_image: Tensor = self.generator(latent)
        score_fake: Tensor = self.discriminator(generated_image)
        score_real: Tensor = self.discriminator(real_images)

        loss_fake = gan_loss(score_fake, False)
        loss_real = gan_loss(score_real, True)
        loss = loss_fake + loss_real

        self.fid.update(real_images, True)
        self.fid.update(generated_image, False)
        return {
            "val/loss": loss,
            "val/score_real": score_real.mean(),
            "val/score_fake": score_fake.mean(),
        }

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
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr
        )
        # discriminator first, generator second
        return (
            {"optimizer": discriminator_optimizer},
            {"optimizer": generator_optimizer},
        )


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=str,
                       help="Input Datasets Directory")
    group.add_argument("--dataset", "-d", type=str,
                       help="Dataset Name predefined in torchvision")
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument("--epoch", "-e", type=int, default=-1)
    step_group.add_argument("--step", "-s", type=int, default=-1)
    parser.add_argument("--checkpoint", "-ckpt", type=str, default=None)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--latent_dim", "-l", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4)
    args = vars(parser.parse_args())
    # fmt: on

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )

    if args["input"]:
        datasets = get_train_test_val_datasets(
            root=args["input"],
            transform=transform,
            loader_mode="L",  # "L" for GrayScale
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
        pin_memory=True,
    )

    sample_image: Tensor = next(iter(train_dataloader))[0]  # (N, C, H, W)
    channels, resolution = sample_image.shape[1:3]

    gan = GAN(
        latent_dim=args["latent_dim"],
        resolution=resolution,
        channels=channels,
        learning_rate=args["learning_rate"],
    )
    callbacks = [
        ModelCheckpoint(save_last_k=3),
        LatentImageSampler(
            step=100, on_epoch_end=False, add_fixed_samples=True
        ),
        LatentDimInterpolator(),
    ]
    gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=args["epoch"],
        max_steps=args["step"],
        check_val_every_n_epoch=10,
        callbacks=callbacks,
        strategy="ddp" if gpus > 1 else None,
    )
    trainer.fit(
        gan,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader or test_dataloader,
    )


if __name__ == "__main__":
    main()
