from typing import List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor

from firewood.common.types import FLOAT, INT
from firewood.trainer.callbacks.image import _ImageCallback
from firewood.utils import search_attr
from firewood.utils.image import batch_flat_to_square


class LatentDimInterpolator(_ImageCallback):
    """
    LatentDimInterpolator

    Unlike the original implementation, store output images on the CPU.
    """

    def __init__(
        self,
        step: Optional[int] = None,
        epoch: int = 1,
        latent_range: FLOAT = (-5.0, 5.0),
        nrow: int = 11,
        ndim_to_interpolate: int = 2,
        padding: int = 0,
        normalize: bool = True,
        norm_range: Tuple[int, int] = (-1, 1),
        on_epoch_end: bool = True,
        scale_each: bool = False,
        pad_value: int = 0,
        save_image: bool = False,
        grid_max_resolution: Optional[int] = None,
    ):
        super().__init__(
            step=step,
            epoch=epoch,
            num_samples=1,
            nrow=nrow,
            padding=padding,
            normalize=normalize,
            norm_range=norm_range,
            on_epoch_end=on_epoch_end,
            add_fixed_samples=False,
            scale_each=scale_each,
            pad_value=pad_value,
            save_image=save_image,
            grid_max_resolution=grid_max_resolution,
        )
        if isinstance(latent_range, (int, float)):
            latent_range = (-latent_range, latent_range)
        elif len(latent_range) == 1:
            latent_range = (-latent_range[0], latent_range[0])
        elif len(latent_range) != 2:
            raise ValueError(
                "`latent_range` should be a scalar or a tuple of length 2. "
                f"Received: {latent_range}"
            )
        self.latent_range = cast(
            Tuple[float, float], tuple(sorted(latent_range))
        )
        self.ndim_to_interpolate = max(ndim_to_interpolate, 2)

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.current_epoch % self.epoch != 0:
            return
        self.forward(trainer, pl_module, title="interpolation/epoch")

    def on_batch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.step is None:
            return
        if trainer.global_step == 0 or trainer.global_step % self.step != 0:
            return
        self.forward(trainer, pl_module, title="interpolation/batch")

    @torch.no_grad()
    def forward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        title: Optional[str] = None,
    ) -> None:
        title = title or "interpolation"
        latent_dim = getattr(pl_module.hparams, "latent_dim", None)
        if latent_dim is None:
            raise ValueError("latent_dim is not set in the model.")
        ndim_to_interpolate = min(self.ndim_to_interpolate, latent_dim)
        images = []

        pl_module.eval()
        for z_x in np.linspace(*self.latent_range, num=self.nrow):
            for z_y in np.linspace(*self.latent_range, num=self.nrow):
                # set all dims to zero
                z = torch.zeros(1, latent_dim, device=pl_module.device)

                # set the dim to interpolate

                div, mod = divmod(ndim_to_interpolate, 2)
                z[:, : div + mod] = torch.tensor(z_x)
                z[:, -div:] = torch.tensor(z_y)

                # sample
                # generate images
                img: Tensor = pl_module(z)

                if len(img.size()) == 2:
                    img_dim = getattr(pl_module, "img_dim", None)
                    if img_dim is None:
                        img = batch_flat_to_square(img)
                    else:
                        img = img.view(-1, *img_dim)

                images.append(img.detach().cpu())
        pl_module.train()

        images = torch.cat(images, dim=0)
        grid = self._make_grid(images)
        self.log_image(trainer, pl_module, grid, title=title)


class ConditionInterpolator(_ImageCallback):
    conditions_base: Optional[Tensor]

    def __init__(
        self,
        step: Optional[int] = None,
        epoch: int = 1,
        conditions_base: Optional[Union[FLOAT, Tensor]] = None,
        conditions_range: FLOAT = (0, 1),
        target_dims: Union[
            INT, range, Sequence[Union[range, Tuple[int, ...]]]
        ] = 0,
        nrow: int = 11,
        padding: int = 0,
        normalize: bool = True,
        norm_range: Tuple[int, int] = (-1, 1),
        on_epoch_end: bool = True,
        scale_each: bool = False,
        pad_value: int = 0,
        save_image: bool = False,
        grid_max_resolution: Optional[int] = None,
    ):
        super().__init__(
            step=step,
            epoch=epoch,
            num_samples=1,
            nrow=nrow,
            padding=padding,
            normalize=normalize,
            norm_range=norm_range,
            on_epoch_end=on_epoch_end,
            add_fixed_samples=False,
            scale_each=scale_each,
            pad_value=pad_value,
            save_image=save_image,
            grid_max_resolution=grid_max_resolution,
        )
        if conditions_base is not None:
            if isinstance(conditions_base, Tensor):
                conditions_base = conditions_base.detach()
            else:
                conditions_base = torch.tensor(conditions_base)
            self.conditions_base = conditions_base.cpu().float().reshape(1, -1)
        else:
            self.conditions_base = None
        if isinstance(conditions_range, (int, float)):
            conditions_range = (-conditions_range, conditions_range)
        elif len(conditions_range) == 1:
            conditions_range = (-conditions_range[0], conditions_range[0])
        elif len(conditions_range) != 2:
            raise ValueError(
                "`conditions_range` should be a scalar or a tuple of length 2. "
                f"Received: {conditions_range}"
            )
        self.conditions_range = cast(
            Tuple[float, float], tuple(sorted(conditions_range))
        )
        _target_dims: List[int] = []
        if isinstance(target_dims, range):
            _target_dims = list(target_dims)
        elif isinstance(target_dims, int):
            _target_dims = [target_dims]
        else:
            for i in target_dims:
                if isinstance(i, int):
                    _target_dims.append(i)
                else:
                    _target_dims.extend(i)
        self.target_dims = tuple(sorted(_target_dims))

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.current_epoch % self.epoch != 0:
            return
        self.forward(trainer, pl_module, title="interpolation/epoch")

    def on_batch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.step is None:
            return
        if trainer.global_step == 0 or trainer.global_step % self.step != 0:
            return
        self.forward(trainer, pl_module, title="interpolation/batch")

    @torch.no_grad()
    def forward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        title: Optional[str] = None,
    ) -> None:
        title = title or "interpolation"
        if self.conditions_base is None:
            condition_dim = search_attr(
                pl_module.hparams,
                ["condition_dim", "n_conditions", "latent_dim"],
            )
            if condition_dim is None:
                raise ValueError("`condition_dim` is not set in the model.")
        else:
            condition_dim = None
        images = []

        pl_module.eval()
        for z_x in np.linspace(*self.conditions_range, num=self.nrow):
            for z_y in np.linspace(*self.conditions_range, num=self.nrow):
                # set all dims to zero
                if self.conditions_base is None:
                    z = torch.zeros(
                        size=(1, condition_dim),
                        device=pl_module.device,
                        pin_memory=True,
                    )
                else:
                    z = self.conditions_base.to(
                        device=pl_module.device, non_blocking=True
                    )

                # set the dim to interpolate
                div, mod = divmod(len(self.target_dims), 2)
                target_x = self.target_dims[: div + mod]
                target_y = self.target_dims[-div:]
                z[:, target_x] = torch.tensor(z_x, dtype=z.dtype)
                z[:, target_y] = torch.tensor(z_y, dtype=z.dtype)

                # sample
                # generate images
                img: Tensor = pl_module(z)

                if len(img.size()) == 2:
                    img_dim = getattr(pl_module, "img_dim", None)
                    if img_dim is None:
                        img = batch_flat_to_square(img)
                    else:
                        img = img.view(-1, *img_dim)

                images.append(img.detach().cpu())
        pl_module.train()

        images = torch.cat(images, dim=0)
        grid = self._make_grid(images)
        self.log_image(trainer, pl_module, grid, title=title)
