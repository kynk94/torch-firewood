import argparse
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import Tensor
from torchvision import transforms

from firewood.common.backend import set_runtime_build
from firewood.common.types import NUMBER
from firewood.models.gan.StyleGAN2 import Discriminator, Generator
from firewood.trainer.callbacks import (
    ExponentialMovingAverage,
    LatentDimInterpolator,
    LatentImageSampler,
    ModelCheckpoint,
)
from firewood.trainer.metrics import (
    FrechetInceptionDistance,
    PathLengthPenalty,
    logistic_nonsaturating_loss,
    simple_gradient_penalty,
)
from firewood.trainer.schedulers import PhaseScheduler
from firewood.trainer.utils import find_checkpoint, get_maximum_multiple_batch
from firewood.trainer.utils.data import (
    DataModule,
    NoClassImageFolder,
    get_train_val_test_datasets,
    torchvision_train_val_test_datasets,
)


class StyleGAN2(pl.LightningModule):
    def __init__(
        self,
        latent_dim: int = 512,
        label_dim: int = 0,
        style_dim: int = 512,
        max_channels: int = 512,
        activation: str = "lrelu",
        noise: str = "normal",
        fir: NUMBER = [1, 3, 3, 1],
        mbstd_group: int = 4,
        truncation: float = 0.5,
        resolution: int = 1024,
        image_channels: int = 3,
        dataset_size: int = 60000,
        initial_batch_size: int = 64,
        lr_equalization: bool = True,
        learning_rate: float = 2e-4,
        r1_gamma: float = 10.0,
        r2_gamma: float = 0.0,
        pl_penalty_gamma: float = 2.0,
        g_reg_interval: int = 4,
        d_reg_interval: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator(
            latent_dim=latent_dim,
            label_dim=label_dim,
            style_dim=style_dim,
            max_channels=max_channels,
            activation=activation,
            noise=noise,
            fir=fir,
            resolution=resolution,
            image_channels=image_channels,
            style_mixing_probability=0.9,
            lr_equalization=lr_equalization,
            mapping_lr_multiplier=0.01,
            synthesis_lr_multiplier=1.0,
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
        self.pl = PathLengthPenalty()

    def forward(
        self,
        input: Tensor,
        truncation: Optional[float] = None,
        resolution: Optional[int] = None,
    ) -> Tensor:
        return self.generator(
            input,
            truncation=truncation or self.hparams.truncation,
            resolution=resolution,
        )

    def generate_latent(self, batch_size: int) -> Tensor:
        return torch.randn(
            size=(batch_size, self.hparams.latent_dim), device=self.device
        )

    def generator_step(
        self,
        input: Tensor,
        truncation: float = 0.5,
    ) -> Dict[str, Tensor]:
        latent = self.generate_latent(input.size(0))
        generated_image: Tensor = self.generator(latent, truncation=truncation)
        score_fake: Tensor = self.discriminator(generated_image)
        loss = logistic_nonsaturating_loss(score_fake, True)
        log_dict = {"loss/gen": loss}
        if (
            (self.global_step // 2) % self.hparams.g_reg_interval == 0
            and self.hparams.pl_penalty_gamma > 0
        ):
            batch_size = max(input.size(0) // 2, 1)
            latent = self.generate_latent(batch_size)
            style = self.generator.make_style(latent, truncation=truncation)
            generated_image = self.generator.synthesis(style)
            pl_penalty = self.pl(style, generated_image)
            pl_penalty *= self.hparams.pl_penalty_gamma
            loss += pl_penalty
            log_dict.update({"loss/gen_pl_penalty": pl_penalty})
        return log_dict

    def discriminator_step(
        self, input: Tensor, truncation: float = 0.5
    ) -> Dict[str, Tensor]:
        latent = self.generate_latent(input.size(0))
        with torch.no_grad():
            generated_image: Tensor = self.generator(
                latent, truncation=truncation
            )
        if self.hparams.r1_gamma != 0.0:
            input.requires_grad = True
        score_real: Tensor = self.discriminator(input)
        score_fake: Tensor = self.discriminator(generated_image)

        loss_real = logistic_nonsaturating_loss(score_real, True)
        loss_fake = logistic_nonsaturating_loss(score_fake, False)
        loss = loss_real + loss_fake

        log_dict = dict()
        if (self.global_step // 2) % self.hparams.d_reg_interval == 0:
            if self.hparams.r1_gamma != 0.0:
                r1_penalty = simple_gradient_penalty(score_real, input)
                r1_penalty *= self.hparams.r1_gamma / 2
                loss += r1_penalty
                log_dict.update({"loss/dis_r1_penalty": r1_penalty})
            if self.hparams.r2_gamma != 0.0:
                r2_penalty = simple_gradient_penalty(
                    score_fake, generated_image
                )
                r2_penalty *= self.hparams.r2_gamma / 2
                loss += r2_penalty
                log_dict.update({"loss/dis_r2_penalty": r2_penalty})

        log_dict.update(
            {
                "loss/dis": loss,
                "loss/dis_real": loss_real,
                "loss/dis_fake": loss_fake,
                "score/real": score_real.mean(),
                "score/fake": score_fake.mean(),
            }
        )
        return log_dict

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> None:
        real_images, _ = batch
        real_images = get_maximum_multiple_batch(
            input=real_images, divisor=self.hparams.mbstd_group
        )

        optimizers = self.optimizers()
        for optimizer_idx in range(len(optimizers)):
            if optimizer_idx == 0:
                log_dict = self.discriminator_step(
                    real_images, self.hparams.truncation
                )
                key = "loss/dis"
            else:
                log_dict = self.generator_step(
                    real_images, self.hparams.truncation
                )
                key = "loss/gen"
            optimizers[optimizer_idx].zero_grad()
            loss = log_dict.pop(key)
            self.manual_backward(loss)
            optimizers[optimizer_idx].step()
            self.log(key, loss, prog_bar=True)
            self.log_dict(log_dict)
        self.update_scheduler(real_images.size(0))

    def validation_step(
        self, batch: Tensor, batch_idx: int
    ) -> Dict[str, Tensor]:
        real_images, _ = batch
        real_images = get_maximum_multiple_batch(
            input=real_images, divisor=self.hparams.mbstd_group
        )

        latent = self.generate_latent(real_images.size(0))
        generated_image: Tensor = self.generator(
            latent, truncation=self.hparams.truncation
        )
        score_fake: Tensor = self.discriminator(generated_image)
        score_real: Tensor = self.discriminator(real_images)

        loss_fake = logistic_nonsaturating_loss(score_fake, False)
        loss_real = logistic_nonsaturating_loss(score_real, True)
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
            self.generator.parameters(), lr=lr, betas=(0.0, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.0, 0.999)
        )

        max_phase = int(
            math.log2(
                self.hparams.resolution
                / self.generator.synthesis.initial_resolution
            )
        )
        scheduler_kwargs = dict(
            dataset_size=getattr(self.hparams, "dataset_size", 60000),
            max_phase=max_phase,
            fade_epoch=getattr(self.hparams, "fade_epoch", 10.0),
            level_epoch=getattr(self.hparams, "level_epoch", 10.0),
            ramp_up_epoch=getattr(self.hparams, "ramp_up_epoch", 0.0),
            lr_dict=None,
            reset_optimizer=True,
        )
        generator_scheduler = PhaseScheduler(
            generator_optimizer, **scheduler_kwargs
        )
        discriminator_scheduler = PhaseScheduler(
            discriminator_optimizer, **scheduler_kwargs
        )

        return (
            {
                "optimizer": discriminator_optimizer,
                "lr_scheduler": {
                    "scheduler": discriminator_scheduler,
                    "name": "scheduler/dis",
                },
            },
            {
                "optimizer": generator_optimizer,
                "lr_scheduler": {
                    "scheduler": generator_scheduler,
                    "name": "scheduler/gen",
                },
            },
        )

    def update_scheduler(self, batch_size: int) -> None:
        total_batch_size = self.trainer.num_devices * batch_size
        d_scheduler, g_scheduler = self.lr_schedulers()
        d_scheduler.step(total_batch_size)
        g_scheduler.step(total_batch_size)

    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()


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
    parser.add_argument("--resolution", "-r", type=int, default=1024)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--latent_dim", "-l", type=int, default=512)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--runtime_build", "-rb", action="store_true")
    args = vars(parser.parse_args())
    # fmt: on

    pl.seed_everything(0)

    if args["runtime_build"]:
        set_runtime_build(True)

    transform = []
    if args["resolution"] is not None:
        transform.append(transforms.Resize(args["resolution"]))
    transform.extend(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5, inplace=True),
        ]
    )
    transform = transforms.Compose(transform)

    if args["input"]:
        datasets = get_train_val_test_datasets(
            root=args["input"],
            dataset_class=NoClassImageFolder,
            transform=transform,
            loader_mode="RGB",
            split=(60000, 10000),
        )
    else:
        datasets = torchvision_train_val_test_datasets(
            name=args["dataset"], root="./datasets", transform=transform
        )
    datamodule = DataModule(
        datasets=datasets,
        batch_size=args["batch_size"],
        num_workers=4,
        pin_memory=False,
    )

    model = StyleGAN2(
        latent_dim=512,
        label_dim=0,
        style_dim=512,
        max_channels=512,
        activation="lrelu",
        noise="normal",
        fir=[1, 3, 3, 1],
        truncation=0.5,
        mbstd_group=4,
        resolution=args["resolution"],
        image_channels=3,
        dataset_size=len(datasets[0]),
        initial_batch_size=args["batch_size"],
        lr_equalization=True,
        learning_rate=args["learning_rate"],
        r1_gamma=10.0,
        r2_gamma=0.0,
        pl_penalty_gamma=2.0,
        g_reg_interval=4,
        d_reg_interval=16,
    )

    gpus = torch.cuda.device_count()
    callbacks = [
        ExponentialMovingAverage(target_modules="generator"),
        ModelCheckpoint(save_last_k=3),
        LatentImageSampler(step=500, log_fixed_batch=True, save_image=True),
        LatentDimInterpolator(
            ndim_to_interpolate=args["latent_dim"] // 10, save_image=True
        ),
        LearningRateMonitor(),
    ]
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        max_epochs=args["epoch"],
        max_steps=args["step"],
        precision=32,
        check_val_every_n_epoch=5,
        callbacks=callbacks,
        strategy="ddp" if gpus > 1 else None,
        log_every_n_steps=10,
    )
    trainer.logger._default_hp_metric = False

    ckpt_path = (
        find_checkpoint(args["checkpoint"]) if args["checkpoint"] else None
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
