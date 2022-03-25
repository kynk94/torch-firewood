import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import albumentations as A
import albumentations.pytorch.transforms as AT
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

from firewood.common.backend import set_runtime_build
from firewood.models.gan.pix2pix import Generator, PatchGAN
from firewood.trainer.callbacks import I2ISampler, ModelCheckpoint
from firewood.trainer.losses import gan_loss
from firewood.trainer.metrics import FrechetInceptionDistance
from firewood.trainer.utils import find_checkpoint
from firewood.utils import highest_power_of_2
from firewood.utils.data import (
    PairedImageFolder,
    get_dataloaders,
    get_train_test_val_datasets,
    torchvision_train_test_val_datasets,
)


class Pix2Pix(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        learning_rate: float = 0.0002,
        weight_reconstruction: float = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.discriminator = PatchGAN(
            in_channels=in_channels + out_channels, n_layers=6
        )

        # metrics
        self.fid = FrechetInceptionDistance()

    def forward(self, input: Tensor) -> Tensor:
        return self.generator(input)

    def generator_step(
        self, source_images: Tensor, target_images: Tensor
    ) -> Dict[str, Tensor]:
        generated_image = self.generator(source_images)
        score_fake = self.discriminator(source_images, generated_image)

        loss_adv = gan_loss(score_fake, True)
        loss_l1 = F.l1_loss(generated_image, target_images)
        loss_l1 *= self.hparams.weight_reconstruction

        loss = loss_adv + loss_l1
        return {
            "loss/gen": loss,
            "loss/gen_adv": loss_adv,
            "loss/gen_l1": loss_l1,
        }

    def discriminator_step(
        self, source_images: Tensor, target_images: Tensor
    ) -> Dict[str, Tensor]:
        with torch.no_grad():
            generated_image = self.generator(source_images)
        score_fake: Tensor = self.discriminator(source_images, generated_image)
        score_real: Tensor = self.discriminator(source_images, target_images)

        loss_fake = gan_loss(score_fake, False)
        loss_real = gan_loss(score_real, True)

        loss = loss_fake + loss_real
        return {
            "loss/dis": loss,
            "loss/dis_adv_fake": loss_fake,
            "loss/dis_adv_real": loss_real,
            "score/real": score_real.mean(),
            "score/fake": score_fake.mean(),
        }

    def training_step(
        self, batch: Tensor, batch_idx: int, optimizer_idx: int
    ) -> Tensor:
        source_images, target_images = batch
        if optimizer_idx == 0:
            log_dict = self.discriminator_step(source_images, target_images)
            key = "loss/dis"
        else:
            log_dict = self.generator_step(source_images, target_images)
            key = "loss/gen"
        loss = log_dict.pop(key)
        self.log(key, loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tensor, batch_idx: int
    ) -> Dict[str, Tensor]:
        source_images, target_images = batch
        generated_image = self.generator(source_images)
        score_fake = self.discriminator(source_images, generated_image)
        score_real = self.discriminator(source_images, target_images)

        loss_fake = gan_loss(score_fake, False)
        loss_real = gan_loss(score_real, True)
        loss = loss_fake + loss_real

        self.fid.update(target_images, True)
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
    parser.add_argument("--resolution", "-r", type=int, default=286)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--latent_dim", "-l", type=int, default=100)
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-4)
    parser.add_argument("--runtime_build", "-rb", action="store_true")
    args = vars(parser.parse_args())
    # fmt: on

    if args["runtime_build"]:
        set_runtime_build(True)

    crop_resolution = highest_power_of_2(args["resolution"])
    if args["input"]:
        transform = [A.Resize(args["resolution"], args["resolution"])]
        if crop_resolution < args["resolution"]:
            transform.append(A.RandomCrop(crop_resolution, crop_resolution))
        transform.extend([A.Normalize(0.5, 0.5), AT.ToTensorV2()])
        transform = A.ReplayCompose(transform)
        datasets = get_train_test_val_datasets(
            root=args["input"],
            dataset_class=PairedImageFolder,
            transform=transform,
            loader_mode="RGB",
            split="train/val",
            use_albumentations=True,
        )
    else:
        transform = [transforms.Resize(args["resolution"], antialias=True)]
        if crop_resolution < args["resolution"]:
            transform.append(
                transforms.RandomCrop(crop_resolution, padding_mode="reflect")
            )
        transform.extend(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        )
        transform = transforms.Compose(transform)
        datasets = torchvision_train_test_val_datasets(
            name=args["dataset"], root="./datasets", transform=transform
        )
    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
        datasets=datasets,
        batch_size=args["batch_size"],
        shuffle=True,
        pin_memory=True,
    )

    sample_source, sample_target = next(iter(train_dataloader))
    print(f"Source: {sample_source.shape},\tTarget: {sample_target.shape}")
    in_channels = sample_source.shape[1]
    out_channels = sample_target.shape[1]

    if args["checkpoint"] is not None:
        pix2pix = Pix2Pix.load_from_checkpoint(
            find_checkpoint(args["checkpoint"])
        )
    else:
        pix2pix = Pix2Pix(
            in_channels=in_channels,
            out_channels=out_channels,
            learning_rate=2e-4,
            weight_reconstruction=1e2,
        )

    callbacks = [
        ModelCheckpoint(save_last_k=3),
        I2ISampler(step=500, add_fixed_samples=True),
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
        pix2pix,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader or test_dataloader,
    )


if __name__ == "__main__":
    main()
