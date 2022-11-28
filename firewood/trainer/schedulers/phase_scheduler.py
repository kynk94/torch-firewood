from typing import Dict, Optional, Tuple

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from firewood.trainer.utils import reset_optimizers


class PhaseScheduler(_LRScheduler):
    """
    Management Phase of Progressive GANs.
    Scheduler for progressive increasing resolution of generated images.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        dataset_size: int = 60000,  # 60k train images in FFHQ, total 70k
        max_phase: int = 8,  # log2(1024 / 4) = 8
        fade_epoch: float = 10.0,
        level_epoch: float = 10.0,
        ramp_up_epoch: float = 0.0,
        lr_dict: Optional[Dict[int, float]] = None,
        reset_optimizer: bool = True,
        last_epoch: int = -1,
    ) -> None:
        self.dataset_size = dataset_size
        self.max_phase = max_phase
        self.fade_epoch = fade_epoch
        self.level_epoch = level_epoch
        self.ramp_up_epoch = ramp_up_epoch
        self.lr_dict = lr_dict or dict()
        self.reset_optimizer = reset_optimizer

        self.viewed_data = 0
        self.phase = 0
        self.alpha = 1.0

        super().__init__(optimizer, last_epoch)

    def step(self, viewed_data: Optional[int] = None) -> None:
        if viewed_data is not None:
            self.update(viewed_data)
        super().step()

    def update(self, viewed_data: int) -> None:
        """
        Update the scheduler state.

        Unlike the official implementation, intuitively designed to increase
        the alpha from 0 to 1 during fade epoch.

        viewed_data:   new phase         fade_epoch done        level_epoch done
        alpha:      increase from 0 ----------- 1 -------------------- 1
        resolution:        n  ----------------- n ------------------ 2 * n
        """
        self.viewed_data += viewed_data

        data_per_fade = int(self.dataset_size * self.fade_epoch)
        data_per_level = int(self.dataset_size * self.level_epoch)
        data_per_phase = data_per_fade + data_per_level
        phase, phase_viewed_data = divmod(self.viewed_data, data_per_phase)

        if phase == 0:
            alpha = 1.0
        elif phase <= self.max_phase:
            alpha = min(phase_viewed_data / data_per_fade, 1.0)
        else:
            alpha = 1.0
            phase = self.max_phase

        if self.reset_optimizer and self.phase != phase:
            reset_optimizers(self.optimizer)

        self.phase = phase
        self.alpha = alpha

    def get_lr(self) -> Tuple[float, ...]:  # type: ignore
        lr = tuple(
            self.lr_dict.get(self.phase, param_group["initial_lr"])
            for param_group in self.optimizer.param_groups
        )
        if self.ramp_up_epoch <= 0.0:
            return lr

        ramp_up_key_data = int(self.dataset_size * self.ramp_up_epoch)
        ramp_up = min(self.viewed_data / ramp_up_key_data, 1.0)
        return tuple(lr_ * ramp_up for lr_ in lr)
