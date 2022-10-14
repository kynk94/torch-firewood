from typing import Dict, Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

DEFAULT_LR = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}


class ProgressiveScheduler(_LRScheduler):
    """
    Scheduler for progressive increasing resolution of generated images.
    """

    optimizer: Optimizer

    def __init__(
        self,
        optimizer: Optimizer,
        dataset_size: int = 60000,  # 60k train images in FFHQ, total 70k
        initial_resolution: int = 4,
        max_resolution: int = 1024,
        level_epoch: float = 10.0,
        fade_epoch: float = 10.0,
        ramp_up_epoch: float = 0.0,
        lr_dict: Dict[int, float] = DEFAULT_LR,
        last_epoch: int = -1,
    ) -> None:
        self.dataset_size = dataset_size
        self.initial_resolution = initial_resolution
        self.max_resolution = max_resolution
        self.level_epoch = level_epoch
        self.fade_epoch = fade_epoch
        self.ramp_up_epoch = ramp_up_epoch
        self.lr_dict = lr_dict

        self.viewed_key_images = 0
        self.phase = 0
        self.alpha = 1.0
        self.resolution = self.initial_resolution

        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def update(self, viewed_key_images: int) -> None:
        """
        key_images: new phase            level_epoch                  fade_epoch
        alpha:      1 ----------------- decrease start ----------------------- 0
        """
        self.viewed_key_images += viewed_key_images
        if self.resolution == self.max_resolution:
            return

        level_key_images = int(self.dataset_size * self.level_epoch)
        fade_key_images = int(self.dataset_size * self.fade_epoch)
        phase_key_images = level_key_images + fade_key_images
        phase, phase_viewed_images = divmod(
            self.viewed_key_images, phase_key_images
        )
        alpha = 1.0
        if phase_viewed_images > level_key_images:
            alpha -= (phase_viewed_images - level_key_images) / fade_key_images

        resolution = min(
            int(self.initial_resolution * 2**phase), self.max_resolution
        )
        if resolution == self.max_resolution:
            alpha = 1.0

        self.phase = phase
        self.alpha = alpha
        self.resolution = resolution

    def get_lr(self) -> Tuple[float, ...]:  # type: ignore
        lr = tuple(
            self.lr_dict.get(self.resolution, param_group["initial_lr"])
            for param_group in self.optimizer.param_groups
        )
        if self.ramp_up_epoch <= 0.0:
            return lr

        ramp_up_key_images = int(self.dataset_size * self.ramp_up_epoch)
        ramp_up = min(self.viewed_key_images / ramp_up_key_images, 1.0)
        return tuple(lr_ * ramp_up for lr_ in lr)
