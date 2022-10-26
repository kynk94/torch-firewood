import math
from typing import Dict, Optional, Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


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
        fade_epoch: float = 10.0,
        level_epoch: float = 10.0,
        ramp_up_epoch: float = 0.0,
        lr_dict: Optional[Dict[int, float]] = None,
        last_epoch: int = -1,
    ) -> None:
        self.dataset_size = dataset_size
        self.initial_resolution = initial_resolution
        self.max_resolution = max_resolution
        self.fade_epoch = fade_epoch
        self.level_epoch = level_epoch
        self.ramp_up_epoch = ramp_up_epoch
        self.lr_dict = lr_dict or dict()

        self.max_phase = int(
            math.log2(self.max_resolution / self.initial_resolution)
        )
        self.viewed_key_images = 0
        self.phase = 0
        self.alpha = 1.0
        self.resolution = self.initial_resolution

        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def update(self, viewed_key_images: int) -> None:
        """
        Update the scheduler state.

        Unlike the official implementation, intuitively designed to increase
        the alpha from 0 to 1 during fade epoch.

        key_images:    new phase         fade_epoch done        level_epoch done
        alpha:      increase from 0 ----------- 1 -------------------- 1
        resolution:        n  ----------------- n ------------------ 2 * n
        """
        self.viewed_key_images += viewed_key_images

        fade_key_images = int(self.dataset_size * self.fade_epoch)
        level_key_images = int(self.dataset_size * self.level_epoch)
        phase_key_images = fade_key_images + level_key_images
        phase, phase_viewed_images = divmod(
            self.viewed_key_images, phase_key_images
        )
        if phase == 0:
            alpha = 1.0
        elif phase > self.max_phase:
            alpha = 1.0
            phase = self.max_phase
        else:
            alpha = min(phase_viewed_images / fade_key_images, 1.0)
        resolution = int(self.initial_resolution * 2**phase)

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
