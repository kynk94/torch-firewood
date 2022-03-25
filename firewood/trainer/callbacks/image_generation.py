import math
from typing import Any, Optional, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from firewood.trainer.callbacks.image import _ImageCallback
from firewood.utils.image import match_channels_between_NCHW_tensor


class LatentImageSampler(_ImageCallback):
    def __init__(
        self,
        step: Optional[int] = None,
        epoch: int = 1,
        num_samples: int = 4,
        nrow: Optional[int] = None,
        padding: int = 0,
        normalize: bool = True,
        norm_range: Tuple[int, int] = (-1, 1),
        on_epoch_end: bool = True,
        add_fixed_samples: bool = False,
        scale_each: bool = False,
        pad_value: int = 0,
        save_image: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            step=step,
            epoch=epoch,
            num_samples=num_samples,
            nrow=nrow,
            padding=padding,
            normalize=normalize,
            norm_range=norm_range,
            on_epoch_end=on_epoch_end,
            add_fixed_samples=add_fixed_samples,
            scale_each=scale_each,
            pad_value=pad_value,
            save_image=save_image,
            **kwargs,
        )

    def forward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        title: Optional[str] = None,
    ) -> None:
        title = title or "images"

        dim = (self.num_samples, getattr(pl_module.hparams, "latent_dim"))
        input = torch.normal(
            mean=0.0, std=1.0, size=dim, device=pl_module.device
        )
        generated_image = self._sample(pl_module, input)
        grid = self._make_grid(generated_image)
        self.log_image(trainer, pl_module, grid, title=title)

        if not self.add_fixed_samples:
            return

        if self.fixed_train_batch is None:
            self.fixed_train_batch = (
                torch.normal(mean=0.0, std=1.0, size=input.shape),
                0,
            )
        fixed_input, _ = self.fixed_train_batch
        fixed_generated_image = self._sample(pl_module, fixed_input)
        fixed_grid = self._make_grid(fixed_generated_image)
        self.log_image(trainer, pl_module, fixed_grid, title=title + "_fixed")

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.forward(trainer, pl_module, title="epoch")

    def on_batch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.step is None:
            return
        if trainer.global_step == 0 or trainer.global_step % self.step != 0:
            return

        self.forward(trainer, pl_module, title="batch")


class ConditionImageSampler(_ImageCallback):
    def __init__(
        self,
        step: Optional[int] = None,
        epoch: int = 1,
        num_samples: int = 4,
        nrow: Optional[int] = None,
        padding: int = 0,
        normalize: bool = True,
        norm_range: Tuple[int, int] = (-1, 1),
        on_epoch_end: bool = True,
        add_fixed_samples: bool = False,
        scale_each: bool = False,
        pad_value: int = 0,
        save_image: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            step=step,
            epoch=epoch,
            num_samples=num_samples,
            nrow=nrow or max(4, math.ceil(math.sqrt(num_samples * 2))),
            padding=padding,
            normalize=normalize,
            norm_range=norm_range,
            on_epoch_end=on_epoch_end,
            add_fixed_samples=add_fixed_samples,
            scale_each=scale_each,
            pad_value=pad_value,
            save_image=save_image,
            **kwargs,
        )

    def forward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        fixed_batch: Optional[Any] = None,
        title: Optional[str] = None,
    ) -> None:
        title = title or "images"
        source_images, conditions = batch[:2]
        source_images: Tensor = source_images[: self.num_samples]
        conditions: Tensor = conditions[: self.num_samples]
        grid = self._batch_to_grid(pl_module, (source_images, conditions))
        self.log_image(trainer, pl_module, grid, title=title)

        if fixed_batch is None:
            return
        self.forward(trainer, pl_module, fixed_batch, title=title + "_fixed")

    def _batch_to_grid(self, pl_module: LightningModule, batch: Any) -> Tensor:
        source_images, conditions = batch[:2]
        source_images = source_images.to(device=pl_module.device)
        conditions = conditions.to(
            dtype=source_images.dtype, device=pl_module.device
        )
        generated_image = self._sample(pl_module, conditions)
        log_source, log_generated = match_channels_between_NCHW_tensor(
            source_images, generated_image
        )
        log_concat = torch.cat([log_source, log_generated], dim=2)
        return self._make_grid(log_concat)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.current_epoch % self.epoch != 0:
            return
        self.forward(
            trainer=trainer,
            pl_module=pl_module,
            batch=self.get_train_batch(trainer),
            fixed_batch=self.get_train_fixed_batch(trainer),
            title="epoch",
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking or trainer.current_epoch % self.epoch != 0:
            return
        self.forward(
            trainer=trainer,
            pl_module=pl_module,
            batch=self.get_val_batch(trainer),
            fixed_batch=self.get_val_fixed_batch(trainer),
            title="val/epoch",
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if self.step is None:
            return
        if trainer.global_step == 0 or trainer.global_step % self.step != 0:
            return
        self.forward(
            trainer=trainer,
            pl_module=pl_module,
            batch=batch,
            fixed_batch=self.get_train_fixed_batch(trainer),
            title="batch",
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if trainer.sanity_checking or self.step is None:
            return
        if trainer.global_step == 0 or trainer.global_step % self.step != 0:
            return
        self.forward(
            trainer=trainer,
            pl_module=pl_module,
            batch=batch,
            fixed_batch=self.get_val_fixed_batch(trainer),
            title="val/batch",
        )


class I2ISampler(_ImageCallback):
    def __init__(
        self,
        step: Optional[int] = None,
        epoch: int = 1,
        num_samples: int = 4,
        nrow: Optional[int] = None,
        padding: int = 0,
        normalize: bool = True,
        norm_range: Tuple[int, int] = (-1, 1),
        on_epoch_end: bool = True,
        add_fixed_samples: bool = False,
        scale_each: bool = False,
        pad_value: int = 0,
        save_image: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            step=step,
            epoch=epoch,
            num_samples=num_samples,
            nrow=nrow or max(4, math.ceil(math.sqrt(num_samples * 3))),
            padding=padding,
            normalize=normalize,
            norm_range=norm_range,
            on_epoch_end=on_epoch_end,
            add_fixed_samples=add_fixed_samples,
            scale_each=scale_each,
            pad_value=pad_value,
            save_image=save_image,
            **kwargs,
        )

    def forward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        fixed_batch: Optional[Any] = None,
        title: Optional[str] = None,
    ) -> None:
        title = title or "images"
        source_images, target_images = batch[:2]
        source_images: Tensor = source_images[: self.num_samples]
        target_images: Tensor = target_images[: self.num_samples]
        grid = self._batch_to_grid(pl_module, (source_images, target_images))
        self.log_image(trainer, pl_module, grid, title=title)

        if fixed_batch is None:
            return
        self.forward(trainer, pl_module, fixed_batch, title=title + "_fixed")

    def _batch_to_grid(self, pl_module: LightningModule, batch: Any) -> Tensor:
        source_images, target_images = batch[:2]
        source_images = source_images.to(device=pl_module.device)
        target_images = target_images.to(device=pl_module.device)
        generated_image = self._sample(pl_module, source_images)
        log_source, log_target = match_channels_between_NCHW_tensor(
            source_images, target_images
        )
        _, log_fake = match_channels_between_NCHW_tensor(
            source_images, generated_image
        )
        log_concat = torch.cat([log_source, log_target, log_fake], dim=2)
        return self._make_grid(log_concat)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.current_epoch % self.epoch != 0:
            return
        self.forward(
            trainer=trainer,
            pl_module=pl_module,
            batch=self.get_train_batch(trainer),
            fixed_batch=self.get_train_fixed_batch(trainer),
            title="epoch",
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking or trainer.current_epoch % self.epoch != 0:
            return
        self.forward(
            trainer=trainer,
            pl_module=pl_module,
            batch=self.get_val_batch(trainer),
            fixed_batch=self.get_val_fixed_batch(trainer),
            title="val/epoch",
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        unused: Optional[int] = 0,
    ) -> None:
        if self.step is None:
            return
        if trainer.global_step == 0 or trainer.global_step % self.step != 0:
            return
        self.forward(
            trainer=trainer,
            pl_module=pl_module,
            batch=batch,
            fixed_batch=self.get_train_fixed_batch(trainer),
            title="batch",
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if trainer.sanity_checking or self.step is None:
            return
        if trainer.global_step == 0 or trainer.global_step % self.step != 0:
            return
        self.forward(
            trainer=trainer,
            pl_module=pl_module,
            batch=batch,
            fixed_batch=self.get_val_fixed_batch(trainer),
            title="val/batch",
        )
