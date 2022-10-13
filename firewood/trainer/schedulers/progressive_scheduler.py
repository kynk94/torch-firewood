import math
from typing import Dict, Optional, Tuple

from pytorch_lightning import Trainer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from firewood.trainer.utils.data import update_train_batch_size_of_trainer

DEFAULT_LR = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}


class ProgressiveScheduler(_LRScheduler):
    """
    Scheduler for progressive increasing resolution of generated images.
    """

    trainer: Trainer
    optimizer: Optimizer

    def __init__(
        self,
        optimizer: Optimizer,
        dataset_size: int = 60000,  # 60k train images in FFHQ, total 70k
        initial_resolution: int = 4,
        max_resolution: int = 1024,
        level_epoch: float = 1.0,
        fade_epoch: float = 1.0,
        ramp_up_epoch: float = 0.0,
        lr_dict: Dict[int, float] = DEFAULT_LR,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)
        self.dataset_size = dataset_size
        self.initial_resolution = initial_resolution
        self.max_resolution = max_resolution
        self.level_epoch = level_epoch
        self.fade_epoch = fade_epoch
        self.ramp_up_epoch = ramp_up_epoch
        self.lr_dict = lr_dict

        self.current_key_images = 0
        self.phase = 0
        self.alpha = 1.0
        self.resolution = initial_resolution
        self.initial_batch_size: Optional[int] = None

    def update(self, batch_size: int) -> None:
        """
        key_images: new phase            level_epoch                  fade_epoch
        alpha:      1 ----------------- decrease start ----------------------- 0
        """
        if self.initial_batch_size is None:
            self.initial_batch_size = batch_size
        self.current_key_images += batch_size
        level_key_images = int(self.dataset_size * self.level_epoch)
        fade_key_images = int(self.dataset_size * self.fade_epoch)
        phase_key_images = level_key_images + fade_key_images
        phase, phase_seen_images = divmod(
            self.current_key_images, phase_key_images
        )
        alpha = 1.0
        if phase_seen_images > level_key_images:
            alpha -= (phase_seen_images - level_key_images) / fade_key_images

        resolution = min(
            int(self.initial_resolution * 2**phase), self.max_resolution
        )
        self.mod_lr(resolution)

        self.phase = phase
        self.alpha = alpha
        self.resolution = resolution

        # If next step is next phase, update batch size of dataset
        if (
            getattr(self, "trainer", None) is not None
            and phase_seen_images + batch_size >= phase_key_images
        ):
            next_batch = max(batch_size // 2, self.initial_batch_size // 16, 2)
            update_train_batch_size_of_trainer(self.trainer, next_batch)

    def mod_lr(self, resolution: int) -> None:
        for param_group in self.optimizer.param_groups:
            lr = self.lr_dict.get(resolution, param_group["initial_lr"])
            param_group["lr"] = lr
        if self.ramp_up_epoch <= 0.0:
            return

        ramp_up_key_images = int(self.dataset_size * self.ramp_up_epoch)
        ramp_up = min(self.current_key_images / ramp_up_key_images, 1.0)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= ramp_up

    def get_lr(self) -> Tuple[float, ...]:  # type: ignore
        return tuple(
            param_group["lr"] for param_group in self.optimizer.param_groups
        )
