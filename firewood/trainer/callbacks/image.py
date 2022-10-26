import math
import os
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, cast

import torch
import torchvision.utils as TU
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.trainer.connectors.data_connector import (
    _DataLoaderSource,
)
from torch import Tensor
from torch.utils.data import DataLoader

from firewood import utils
from firewood.common.types import DEVICE
from firewood.utils.image import batch_flat_to_square, save_tensor_to_image
from firewood.utils.torch_op import args_to, kwargs_to


def _pass_through(*args: Any, **kwargs: Any) -> None:
    return None


class _ImageCallback(Callback):
    def __init__(
        self,
        step: Optional[int] = None,
        epoch: int = 1,
        num_samples: int = 4,
        nrow: Optional[int] = None,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        on_epoch_end: bool = True,
        log_fixed_batch: bool = False,
        sample_fixed_batch: bool = False,
        scale_each: bool = False,
        pad_value: int = 0,
        save_image: bool = False,
        grid_max_resolution: Optional[int] = None,
        sample_args: Optional[dict] = None,
    ) -> None:
        self.step = step
        self.epoch = epoch
        self.num_samples = num_samples
        self.nrow = nrow or max(4, math.ceil(math.sqrt(num_samples)))
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.log_fixed_batch = log_fixed_batch
        self.sample_fixed_batch = sample_fixed_batch
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.save_image = save_image
        self.grid_max_resolution = grid_max_resolution
        self.sample_args = sample_args or dict()
        self.device: Optional[DEVICE] = None

        if self.step is None:
            if not on_epoch_end:
                raise ValueError(
                    "step must be specified if on_epoch_end is False"
                )
            setattr(self, "on_train_batch_end", _pass_through)
            setattr(self, "on_test_batch_end", _pass_through)
            setattr(self, "on_validation_batch_end", _pass_through)
            setattr(self, "on_predict_batch_end", _pass_through)
        if not on_epoch_end:
            setattr(self, "on_train_epoch_end", _pass_through)
            setattr(self, "on_test_epoch_end", _pass_through)
            setattr(self, "on_validation_epoch_end", _pass_through)
            setattr(self, "on_predict_epoch_end", _pass_through)

        # test is not supported
        self._train_dataloader: Optional[DataLoader] = None
        self._val_dataloader: Optional[DataLoader] = None
        self._train_dataloader_iter: Optional[Iterator] = None
        self._val_dataloader_iter: Optional[Iterator] = None
        self._fixed_train_batch: Optional[Any] = None
        self._fixed_val_batch: Optional[Any] = None

    @torch.no_grad()
    def _sample(
        self,
        pl_module: LightningModule,
        input: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        if self.device is None:
            self.device = pl_module.device
        kwargs.update(self.sample_args)
        pl_module.eval()
        generated_images: Tensor = pl_module(
            input.to(device=pl_module.device, non_blocking=True),
            *args_to(*args, dtype=input.dtype, device=self.device),
            **kwargs_to(**kwargs, dtype=input.dtype, device=self.device),
        )
        pl_module.train()

        if generated_images.ndim == 2:
            img_dim = getattr(pl_module, "img_dim", None)
            if img_dim is None:
                generated_images = batch_flat_to_square(generated_images)
            else:
                generated_images = generated_images.view(
                    self.num_samples, *img_dim
                )
        return generated_images

    @torch.no_grad()
    def _make_grid(self, input: Tensor) -> Tensor:
        grid = TU.make_grid(
            tensor=input,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        if (
            self.grid_max_resolution is None
            or max(grid.shape[-2:]) < self.grid_max_resolution
        ):
            return grid
        return utils.image.tensor_resize(
            grid, self.grid_max_resolution, antialias=True
        )

    def log_image(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        input: Tensor,
        title: Optional[str] = None,
        global_step: Optional[int] = None,
    ) -> None:
        if not trainer.is_global_zero:
            return
        title = title or "images"
        str_title = f"{pl_module.__class__.__name__}_{title}"
        global_step = global_step or trainer.global_step

        writer = getattr(trainer.logger, "experiment")
        writer.add_image(str_title, input, global_step=global_step)

        if self.save_image:
            log_dir = getattr(trainer.logger, "log_dir")
            image_dir = utils.makedirs(log_dir, "images")
            basename = utils.validate_filename(f"{str_title}_{global_step}.png")
            save_tensor_to_image(input, os.path.join(image_dir, basename))

    def _reset_dataloader(self, trainer: Trainer) -> None:
        for stage in ("train", "val"):
            source: Optional[_DataLoaderSource] = getattr(
                trainer._data_connector, f"_{stage}_dataloader_source", None
            )
            if source is None:
                continue
            dataloader = source.dataloader()
            if isinstance(dataloader, Mapping):
                dataset = cast(
                    DataLoader, next(iter(dataloader.values()))
                ).dataset
            elif isinstance(dataloader, Sequence):
                dataset = cast(DataLoader, dataloader[0]).dataset
            else:
                dataset = dataloader.dataset
            dataloader = DataLoader(
                dataset,
                batch_size=self.num_samples,
                shuffle=False,
                drop_last=True,
                num_workers=0,
            )
            setattr(self, f"_{stage}_dataloader", dataloader)
            setattr(self, f"_{stage}_dataloader_iter", iter(dataloader))

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._reset_dataloader(trainer)

    @property
    def train_batch(self) -> Tensor:
        if self._train_dataloader_iter is None:
            raise ValueError("Train dataloader is not set.")
        try:
            return next(self._train_dataloader_iter)
        except StopIteration:
            self._train_dataloader_iter = iter(
                cast(DataLoader, self._train_dataloader)
            )
            return self.train_batch

    @property
    def val_batch(self) -> Tensor:
        if self._val_dataloader_iter is None:
            raise ValueError("Validation dataloader is not set.")
        try:
            return next(self._val_dataloader_iter)
        except StopIteration:
            self._val_dataloader_iter = iter(
                cast(DataLoader, self._val_dataloader)
            )
            return self.val_batch

    @property
    def fixed_train_batch(self) -> Optional[Any]:
        if self._fixed_train_batch is None and self.sample_fixed_batch:
            self._fixed_train_batch = self.train_batch
        return args_to(
            self._fixed_train_batch, device=self.device, non_blocking=True
        )

    @fixed_train_batch.setter
    def fixed_train_batch(self, *args: Any) -> None:
        self._fixed_train_batch = args_to(*args, device="cpu")

    @property
    def fixed_val_batch(self) -> Optional[Any]:
        if self._fixed_val_batch is None and self.sample_fixed_batch:
            self._fixed_val_batch = self.val_batch
        return args_to(
            self._fixed_val_batch, device=self.device, non_blocking=True
        )

    @fixed_val_batch.setter
    def fixed_val_batch(self, *args: Any) -> None:
        self._fixed_val_batch = args_to(*args, device="cpu")
