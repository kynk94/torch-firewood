import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torchvision import transforms

from firewood.common.backend import set_runtime_build
from firewood.common.types import INT
from firewood.models.gan.LSGAN import Discriminator, Generator
from firewood.trainer.callbacks import (
    LatentDimInterpolator,
    LatentImageSampler,
    ModelCheckpoint,
)
from firewood.trainer.metrics import FrechetInceptionDistance, lsgan_loss
from firewood.trainer.utils import find_checkpoint
from firewood.trainer.utils.data import (
    NoClassImageFolder,
    get_dataloaders,
    get_train_val_test_datasets,
    torchvision_train_val_test_datasets,
)


class LSGAN(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 1024,
        gen_n_layers: int = 7,
        gen_n_filters: int = 256,
        gen_activation: str = "relu",
        dis_n_layers: int = 4,
        dis_n_filters: int = 64,
        dis_activation: str = "lrelu",
        resolution: INT = 112,
        channels: int = 3,
        learning_rate: float = 2e-4,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.generator = Generator(
            latent_dim=latent_dim,
            n_layer=gen_n_layers,
            n_filter=gen_n_filters,
            activation=gen_activation,
            resolution=resolution,
            channels=channels,
        )
        self.discriminator = Discriminator(
            n_layer=dis_n_layers,
            n_filter=dis_n_filters,
            activation=dis_activation,
            resolution=resolution,
            channels=channels,
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
        loss = lsgan_loss(score_fake, True)
        return {"loss/gen": loss}

    def discriminator_step(self, input: Tensor) -> Dict[str, Tensor]:
        latent = self.generate_latent(input.size(0))
        with torch.no_grad():
            generated_image: Tensor = self.generator(latent)
        score_fake: Tensor = self.discriminator(generated_image)
        loss_fake = lsgan_loss(score_fake, False)

        score_real: Tensor = self.discriminator(input)
        loss_real = lsgan_loss(score_real, True)
        loss = loss_fake + loss_real
        return {
            "loss/dis": loss,
            "score/real": score_real.mean(),
            "score/fake": score_fake.mean(),
        }

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        real_images, _ = batch
        optimizers = self.optimizers()
        for optimizer_idx in range(len(optimizers)):
            if optimizer_idx == 0:
                log_dict = self.discriminator_step(real_images)
                key = "loss/dis"
            else:
                log_dict = self.generator_step(real_images)
                key = "loss/gen"
            optimizers[optimizer_idx].zero_grad()
            loss = log_dict.pop(key)
            self.manual_backward(loss)
            optimizers[optimizer_idx].step()
            self.log(key, loss, prog_bar=True)
            self.log_dict(log_dict)

    def validation_step(
        self, batch: Tensor, batch_idx: int
    ) -> Dict[str, Tensor]:
        real_images, _ = batch

        latent = self.generate_latent(real_images.size(0))
        generated_image = self.generator(latent)
        score_fake = self.discriminator(generated_image)
        score_real = self.discriminator(real_images)

        loss_fake = lsgan_loss(score_fake, False)
        loss_real = lsgan_loss(score_real, True)
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
        self.log_dict(log_dict, sync_dist=True)

    def configure_optimizers(self) -> Tuple[Any]:
        lr = self.hparams.learning_rate
        generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.5, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
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

    pl.seed_everything(0)

    if args["runtime_build"]:
        set_runtime_build(True)

    transform = []
    if args["resolution"] is not None:
        transform.append(transforms.Resize(args["resolution"]))
    transform.extend([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    transform = transforms.Compose(transform)

    if args["input"]:
        datasets = get_train_val_test_datasets(
            root=args["input"],
            dataset_class=NoClassImageFolder,
            transform=transform,
            loader_mode="RGB",
            split="train/val",
        )
    else:
        datasets = torchvision_train_val_test_datasets(
            name=args["dataset"], root="./datasets", transform=transform
        )
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        datasets=datasets,
        batch_size=args["batch_size"],
        shuffle=True,
        pin_memory=False,
    )

    sample_image: Tensor = next(iter(train_dataloader))[0]  # (N, C, H, W)
    channels, resolution = sample_image.shape[1:3]

    lsgan = LSGAN(
        latent_dim=args["latent_dim"],
        gen_n_layers=7,
        gen_n_filters=256,
        gen_activation="lrelu",
        dis_n_layers=4,
        dis_n_filters=64,
        dis_activation="lrelu",
        resolution=resolution,
        channels=channels,
        learning_rate=args["learning_rate"],
    )

    callbacks = [
        ModelCheckpoint(save_last_k=3),
        LatentImageSampler(step=500, log_fixed_batch=True, save_image=True),
        LatentDimInterpolator(
            ndim_to_interpolate=args["latent_dim"] // 10, save_image=True
        ),
    ]
    gpus = torch.cuda.device_count()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        max_epochs=args["epoch"],
        max_steps=args["step"],
        precision=32,
        check_val_every_n_epoch=5,
        callbacks=callbacks,
        strategy="ddp" if gpus > 1 else None,
    )
    trainer.logger._default_hp_metric = False

    ckpt_path = (
        find_checkpoint(args["checkpoint"]) if args["checkpoint"] else None
    )
    trainer.fit(
        lsgan,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader or test_dataloader,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
