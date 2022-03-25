import math
import os
from typing import Any, Iterator, Optional, Tuple, cast

import torch
import torchvision.transforms.functional_tensor as TFT
import torchvision.utils as TU
from pytorch_lightning import Callback, LightningModule, Trainer
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
        add_fixed_samples: bool = False,
        scale_each: bool = False,
        pad_value: int = 0,
        save_image: bool = False,
        grid_max_resolution: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.step = step
        self.epoch = epoch
        self.num_samples = num_samples
        self.nrow = nrow or max(4, math.ceil(math.sqrt(num_samples)))
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.add_fixed_samples = add_fixed_samples
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.save_image = save_image
        self.grid_max_resolution = grid_max_resolution
        self.kwargs = kwargs
        self.device: Optional[DEVICE] = None

        if self.step is None:
            if not on_epoch_end:
                raise ValueError(
                    "step must be specified if on_epoch_end is False"
                )
            setattr(self, "on_batch_end", _pass_through)
            setattr(self, "on_train_batch_end", _pass_through)
            setattr(self, "on_test_batch_end", _pass_through)
            setattr(self, "on_validation_batch_end", _pass_through)
            setattr(self, "on_predict_batch_end", _pass_through)
        if not on_epoch_end:
            setattr(self, "on_epoch_end", _pass_through)
            setattr(self, "on_train_epoch_end", _pass_through)
            setattr(self, "on_test_epoch_end", _pass_through)
            setattr(self, "on_validation_epoch_end", _pass_through)
            setattr(self, "on_predict_epoch_end", _pass_through)

        self._train_data_iter: Optional[Iterator] = None
        self._test_data_iter: Optional[Iterator] = None
        self._val_data_iter: Optional[Iterator] = None
        self._fixed_train_batch: Optional[Tuple[Any, ...]] = None
        self._fixed_test_batch: Optional[Tuple[Any, ...]] = None
        self._fixed_val_batch: Optional[Tuple[Any, ...]] = None

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
        args = args_to(*args, dtype=input.dtype, device=self.device)
        kwargs.update(self.kwargs)
        kwargs = kwargs_to(**kwargs, dtype=input.dtype, device=self.device)

        pl_module.eval()
        generated_images: Tensor = pl_module(
            input.to(device=pl_module.device, non_blocking=True),
            *args,
            **kwargs,
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
        if self.grid_max_resolution is None:
            return grid

        height, width = grid.shape[-2:]
        larger = max(height, width)
        if larger < self.grid_max_resolution:
            return grid

        ratio = self.grid_max_resolution / larger
        new_resolution = (int(height * ratio), int(width * ratio))
        return TFT.resize(grid, new_resolution, antialias=True)

    def log_image(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        input: Tensor,
        title: Optional[str] = None,
        global_step: Optional[int] = None,
    ) -> None:
        title = title or "images"
        str_title = f"{pl_module.__class__.__name__}_{title}"
        global_step = global_step or trainer.global_step

        writer = getattr(trainer.logger, "experiment")
        writer.add_image(str_title, input, global_step=global_step)

        if self.save_image and trainer.global_rank == 0:
            log_dir = getattr(trainer.logger, "log_dir")
            image_dir = utils.makedirs(log_dir, "images")
            basename = utils.validate_filename(f"{str_title}_{global_step}.png")
            save_tensor_to_image(input, os.path.join(image_dir, basename))

    def _set_data_iter(self, trainer: Trainer, stage: str = "train") -> None:
        datamodule = getattr(trainer, "datamodule", None)
        stage = stage.lower()
        if datamodule is not None:
            if stage.startswith("train"):
                dataloader: DataLoader = datamodule.train_dataloader
                self._train_data_iter = iter(dataloader)
            elif stage.startswith("test"):
                dataloader = datamodule.test_dataloader
                self._test_data_iter = iter(dataloader)
            elif stage.startswith("val"):
                dataloader = datamodule.val_dataloader
                self._val_data_iter = iter(dataloader)
            else:
                raise ValueError(f"Unknown stage: {stage}")
            return
        if stage.startswith("train"):
            data_source = trainer._data_connector._train_dataloader_source
            dataloader = cast(DataLoader, data_source.instance)
            if dataloader is not None:
                self._train_data_iter = iter(dataloader)
                return
        elif stage.startswith("test"):
            data_source = trainer._data_connector._test_dataloader_source
            dataloader = cast(DataLoader, data_source.instance)
            if dataloader is not None:
                self._test_data_iter = iter(dataloader)
                return
        elif stage.startswith("val"):
            data_source = trainer._data_connector._val_dataloader_source
            dataloader = cast(DataLoader, data_source.instance)
            if dataloader is not None:
                self._val_data_iter = iter(dataloader)
                return
        else:
            raise ValueError(f"Unknown stage: {stage}")
        raise ValueError(
            f"No {stage} dataloader found for {utils.get_name*(trainer)}."
        )

    def get_train_batch(self, trainer: Trainer) -> Any:
        if self._train_data_iter is None:
            self._set_data_iter(trainer, "train")
        return next(cast(Iterator, self._train_data_iter))

    def get_test_batch(self, trainer: Trainer) -> Any:
        if self._test_data_iter is None:
            self._set_data_iter(trainer, "test")
        return next(cast(Iterator, self._test_data_iter))

    def get_val_batch(self, trainer: Trainer) -> Any:
        if self._val_data_iter is None:
            self._set_data_iter(trainer, "validation")
        return next(cast(Iterator, self._val_data_iter))

    def get_train_fixed_batch(self, trainer: Trainer) -> Any:
        if not self.add_fixed_samples:
            return
        if self.fixed_train_batch is None:
            self.fixed_train_batch = self.get_train_batch(trainer)
        return self.fixed_train_batch

    def get_test_fixed_batch(self, trainer: Trainer) -> Any:
        if not self.add_fixed_samples:
            return
        if self.fixed_test_batch is None:
            self.fixed_test_batch = self.get_test_batch(trainer)
        return self.fixed_test_batch

    def get_val_fixed_batch(self, trainer: Trainer) -> Any:
        if not self.add_fixed_samples:
            return
        if self.fixed_val_batch is None:
            self.fixed_val_batch = self.get_val_batch(trainer)
        return self.fixed_val_batch

    @property
    def fixed_train_batch(self) -> Optional[Tuple[Any, ...]]:
        if self._fixed_train_batch is None:
            return None
        if self.device is not None:
            return args_to(*self._fixed_train_batch, device=self.device)
        return self._fixed_train_batch

    @fixed_train_batch.setter
    def fixed_train_batch(self, value: Tuple[Any, ...]) -> None:
        self._fixed_train_batch = args_to(*value, device=self.device)

    @property
    def fixed_test_batch(self) -> Optional[Tuple[Any, ...]]:
        if self._fixed_test_batch is None:
            return None
        if self.device is not None:
            return args_to(*self._fixed_test_batch, device=self.device)
        return self._fixed_test_batch

    @fixed_test_batch.setter
    def fixed_test_batch(self, value: Tuple[Any, ...]) -> None:
        self._fixed_test_batch = args_to(*value, device=self.device)

    @property
    def fixed_val_batch(self) -> Optional[Tuple[Any, ...]]:
        if self._fixed_val_batch is None:
            return None
        if self.device is not None:
            return args_to(*self._fixed_val_batch, device=self.device)
        return self._fixed_val_batch

    @fixed_val_batch.setter
    def fixed_val_batch(self, value: Tuple[Any, ...]) -> None:
        self._fixed_val_batch = args_to(*value, device=self.device)
