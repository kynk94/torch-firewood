import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

from firewood.common.backend import set_runtime_build
from firewood.models.gan.pix2pixHD import (
    Encoder,
    GlobalGenerator,
    LocalEnhancer,
    MultiScalePatchGAN,
)
from firewood.trainer.callbacks import I2ISampler, ModelCheckpoint
from firewood.trainer.losses import PerceptualLoss, lsgan_loss
from firewood.trainer.metrics import FrechetInceptionDistance
from firewood.trainer.utils import extract_state_dict, find_checkpoint
from firewood.utils import highest_power_of_2
from firewood.utils.data import (
    PairedImageFolder,
    get_dataloaders,
    get_train_test_val_datasets,
    torchvision_train_test_val_datasets,
)
from firewood.utils.image import get_semantic_edge, get_semantic_one_hot


class Pix2PixHD(pl.LightningModule):
    encoder: Optional[Encoder] = None

    def __init__(
        self,
        in_channels: int,  # label channels of semantic to image task, 1 or 3
        out_channels: int,  # image channels, mostly 3
        n_filters: int = 64,
        use_one_hot: bool = True,
        n_classes: int = 35,  # number of classes in semantic label
        use_feature_map: bool = False,
        use_edge_map: bool = False,
        feature_map_channels: int = 3,
        feature_map_n_filters: int = 16,
        feature_cluster_path: Optional[str] = None,
        n_discriminators: int = 1,
        n_local_enhancers: int = 0,
        pretrained_global_generator: Optional[str] = None,
        global_train_epoch: int = 0,
        learning_rate: float = 0.0002,
        weight_reconstruction: float = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        dis_in_channels = in_channels
        gen_in_channels = in_channels
        if use_one_hot:
            dis_in_channels = n_classes
            gen_in_channels = n_classes
        if use_feature_map:
            gen_in_channels += feature_map_channels
            self.encoder = Encoder(
                out_channels, feature_map_channels, feature_map_n_filters
            )
        if use_edge_map:
            dis_in_channels += 1
            gen_in_channels += 1

        if n_local_enhancers:
            self.generator = LocalEnhancer(
                in_channels=gen_in_channels,
                out_channels=out_channels,
                n_filters=n_filters // 2**n_local_enhancers,
                n_local_enhancers=n_local_enhancers,
            )
            if pretrained_global_generator is None:
                raise ValueError(
                    "Pretrained global generator is required when n_local_enhancers > 0."
                )
            global_generator_state_dict = torch.load(
                pretrained_global_generator
            )
            self.generator.global_generator.load_state_dict(
                extract_state_dict(
                    global_generator_state_dict, key="generator", pop_key=True
                ),
                strict=False,  # because deleted the last layer
            )
        else:
            self.generator = GlobalGenerator(
                in_channels=gen_in_channels,
                out_channels=out_channels,
                n_filters=n_filters,
            )
        self.discriminator = MultiScalePatchGAN(
            in_channels=dis_in_channels + out_channels,
            n_discriminators=n_discriminators,
        )

        self.perceptual_loss = PerceptualLoss(
            extractor="vgg19",
            targets=[
                "block1_conv1",
                "block2_conv1",
                "block3_conv1",
                "block4_conv1",
                "block5_conv1",
            ],
            weights=[1 / 32, 1 / 16, 1 / 8, 1 / 4, 1],
        )
        # metrics
        self.fid = FrechetInceptionDistance()

    def forward(
        self, source_images: Tensor, instance_map: Optional[Tensor] = None
    ) -> Tensor:
        """
        input: (N, C, H, W), range [-1, 1]
        instance_input: (N, C, H, W), range [0, 255]
        """
        if self.hparams.use_feature_map:
            if instance_map is None:
                raise ValueError("Instance map is required.")
            feature_map = self.sample_feature_map(instance_map)
            generator_input = torch.cat((source_images, feature_map), dim=1)
        else:
            generator_input = source_images
        return self.generator(generator_input)

    def sample_feature_map(self, instance_map: Tensor) -> Tensor:
        clustered = np.load(
            self.hparams.feature_cluster_path, encoding="latin1"
        ).item()
        feature_map_size = list(instance_map.shape)
        feature_map_size[1] = self.hparams.feature_map_channels
        feature_map = torch.zeros(feature_map_size, dtype=instance_map.dtype)

        instance_map = instance_map.cpu().int().numpy()
        for i in np.unique(instance_map):
            label = i if i < 1000 else i // 1000
            if label not in clustered:
                continue

            feature = clustered[label]
            cluster_index = np.random.randint(0, feature.shape[0])
            idx = (instance_map == i).nonzero()
            for k in range(self.hparams.feature_map_channels):
                feature_map[
                    idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]
                ] = feature_map[cluster_index, k]
        return feature_map.to(instance_map.device)

    @torch.no_grad()
    def encode_source_images(
        self,
        source_images: Tensor,  # semantic label
        instance_map: Optional[Tensor] = None,  # instance map of label
    ) -> Tensor:
        """
        source_images: (N, C, H, W), range [-1, 1]
        instance_map: (N, 1, H, W), range [0, ~] long
        """
        if self.hparams.use_one_hot:
            source_images = get_semantic_one_hot(
                input=source_images,
                n_classes=self.hparams.n_classes,
                normalize=True,
                normalize_range=(-1.0, 1.0),
            )
        if self.hparams.use_edge_map:
            if instance_map is None:
                raise ValueError("Instance map is required.")
            edge_map = get_semantic_edge(instance_map)  # (N, 1, H, W), [0, 1]
            source_images = torch.cat((source_images, edge_map), dim=1)
        return source_images

    def generator_step(
        self,
        source_images: Tensor,
        target_images: Tensor,
        instance_map: Optional[Tensor] = None,
        feature_map: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # prepare source images
        source_images = self.encode_source_images(source_images, instance_map)
        if self.hparams.use_feature_map:
            if instance_map is None:
                raise ValueError("Instance map is required.")
            if feature_map in None:
                feature_map = self.encoder(target_images, instance_map, False)
            generator_input = torch.cat((source_images, feature_map), dim=1)
        else:
            generator_input = source_images

        generated_image = self.generator(generator_input)
        features_fake: List[List[Tensor]] = self.discriminator(
            generated_image, source_images, extract_features=True
        )
        with torch.no_grad():
            features_real: List[List[Tensor]] = self.discriminator(
                target_images, source_images, extract_features=True
            )

        loss_adv = 0
        loss_features = 0
        for discriminator_index in range(self.hparams.n_discriminators):
            _feature_fake = features_fake[discriminator_index]
            _feature_real = features_real[discriminator_index]
            for fake, real in zip(_feature_fake, _feature_real):
                loss_adv += lsgan_loss(fake, True)
                loss_features += F.l1_loss(fake, real)

        loss_perceptual = self.perceptual_loss(generated_image, target_images)

        loss = loss_adv + loss_features + loss_perceptual
        return {
            "loss/gen": loss,
            "loss/gen_adv": loss_adv,
            "loss/gen_features": loss_features,
            "loss/gen_perceptual": loss_perceptual,
        }

    def discriminator_step(
        self,
        source_images: Tensor,
        target_images: Tensor,
        instance_map: Optional[Tensor] = None,
        feature_map: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # prepare source images
        source_images = self.encode_source_images(source_images, instance_map)
        if self.hparams.use_feature_map:
            if instance_map is None:
                raise ValueError("Instance map is required.")
            if feature_map in None:
                feature_map = self.encoder(target_images, instance_map, False)
            generator_input = torch.cat((source_images, feature_map), dim=1)
        else:
            generator_input = source_images

        with torch.no_grad():
            generated_image = self.generator(generator_input)
        features_fake = self.discriminator(
            generated_image, source_images, extract_features=True
        )
        features_real = self.discriminator(
            target_images, source_images, extract_features=True
        )

        loss_adv_fake = 0
        loss_adv_real = 0
        for discriminator_index in range(self.hparams.n_discriminators):
            _feature_fake = features_fake[discriminator_index]
            _feature_real = features_real[discriminator_index]
            for fake, real in zip(_feature_fake, _feature_real):
                loss_adv_fake += lsgan_loss(fake, False)
                loss_adv_real += lsgan_loss(real, True)

        loss = loss_adv_fake + loss_adv_real
        return {
            "loss/dis": loss,
            "loss/dis_adv_fake": loss_adv_fake,
            "loss/dis_adv_real": loss_adv_real,
            "score/real": features_real[0][-1].mean(),
            "score/fake": features_fake[0][-1].mean(),
        }

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.hparams.global_train_epoch:
            for param in self.generator.parameters():
                param.requires_grad = True

    def training_step(
        self, batch: Tensor, batch_idx: int, optimizer_idx: int
    ) -> Tensor:
        if len(batch) == 4:
            source_images, target_images, instance_map, feature_map = batch
        else:
            source_images, target_images = batch[:2]
            instance_map, feature_map = None, None
        if optimizer_idx == 0:
            log_dict = self.discriminator_step(
                source_images, target_images, instance_map, feature_map
            )
            key = "loss/dis"
        else:
            log_dict = self.generator_step(
                source_images, target_images, instance_map, feature_map
            )
            key = "loss/gen"
        loss = log_dict.pop(key)
        self.log(key, loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tensor, batch_idx: int
    ) -> Dict[str, Tensor]:
        if len(batch) == 4:
            source_images, target_images, instance_map, feature_map = batch
        else:
            source_images, target_images = batch[:2]
            instance_map, feature_map = None, None
        if self.hparams.use_feature_map:
            if instance_map is None:
                raise ValueError("Instance map is required.")
            feature_map = self.sample_feature_map(instance_map)
            generator_input = torch.cat((source_images, feature_map), dim=1)
        else:
            generator_input = source_images

        generated_image = self.generator(generator_input)
        features_fake = self.discriminator(
            generated_image, source_images, extract_features=True
        )
        features_real = self.discriminator(
            target_images, source_images, extract_features=True
        )

        loss_adv_fake = 0
        loss_adv_real = 0
        for discriminator_index in range(self.hparams.n_discriminators):
            _feature_fake = features_fake[discriminator_index]
            _feature_real = features_real[discriminator_index]
            for fake, real in zip(_feature_fake, _feature_real):
                loss_adv_fake += lsgan_loss(fake, False)
                loss_adv_real += lsgan_loss(real, True)
        loss = loss_adv_fake + loss_adv_real

        self.fid.update(target_images, True)
        self.fid.update(generated_image, False)
        return {
            "val/loss": loss,
            "val/score_real": features_real[0][-1].mean(),
            "val/score_fake": features_fake[0][-1].mean(),
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

    transform = [transforms.Resize(args["resolution"])]
    crop_resolution = highest_power_of_2(args["resolution"])
    if crop_resolution < args["resolution"]:
        transform.append(transforms.RandomCrop(crop_resolution))
    transform.extend(
        [
            transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
            transforms.Normalize(0.5, 0.5),  # range [0.0, 1.0] -> [-1.0, 1.0]
        ]
    )
    transform = transforms.Compose(transform)

    if args["input"]:
        datasets = get_train_test_val_datasets(
            root=args["input"],
            dataset_class=PairedImageFolder,  # TODO: implement instance map support
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
        pin_memory=True,
    )

    sample_source, sample_target = next(iter(train_dataloader))[:2]
    print(f"Source: {sample_source.shape},\tTarget: {sample_target.shape}")
    in_channels = sample_source.shape[1]
    out_channels = sample_target.shape[1]

    if args["checkpoint"] is not None:
        pix2pix_hd = Pix2PixHD.load_from_checkpoint(
            find_checkpoint(args["checkpoint"])
        )
    else:
        pix2pix_hd = Pix2PixHD(
            in_channels=in_channels,
            out_channels=out_channels,
            n_filters=64,
            use_one_hot=False,
            n_classes=35,  # number of classes in semantic label
            use_feature_map=False,
            use_edge_map=False,
            feature_map_channels=3,
            feature_map_n_filters=16,
            feature_cluster_path=None,
            n_discriminators=1,
            n_local_enhancers=0,
            pretrained_global_generator=None,  # Need when train local_enhancer
            global_train_epoch=0,  #  epoch to start fine tune global generator
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
        pix2pix_hd,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader or test_dataloader,
    )


if __name__ == "__main__":
    main()
