from typing import Dict, Optional, Tuple

from pytorch_lightning import Trainer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

DEFAULT_LR = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}


class ProgressiveScheduler(_LRScheduler):
    """
    Scheduler for progressive increasing resolution of generated images.
    """

    trainer: Trainer

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
        last_epoch=-1,
    ) -> None:
        super().__init__(optimizer, last_epoch)
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

    def update(self, batch_size: int) -> Tuple[float, int]:
        """
        key_images: new phase            level_epoch                  fade_epoch
        alpha:      1 ----------------- decrease start ----------------------- 0
        """
        self.current_key_images += batch_size
        level_key_images = int(self.dataset_size * self.level_epoch)
        fade_key_images = int(self.dataset_size * self.fade_epoch)
        phase_key_images = level_key_images + fade_key_images
        phase, current_key_images = divmod(
            self.current_key_images, phase_key_images
        )
        alpha = 1.0
        if current_key_images > level_key_images:
            alpha -= (current_key_images - level_key_images) / fade_key_images

        resolution = min(
            int(self.initial_resolution * 2**phase), self.max_resolution
        )
        self.mod_lr(resolution)

        self.phase = phase
        self.alpha = alpha
        self.resolution = resolution

        # If next step is next phase, update batch size of dataset
        if self.current_key_images + batch_size >= phase_key_images:
            self.mod_batch_size_of_dataset()

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

    def mod_batch_size_of_dataset(self) -> None:
        trainer: Optional[Trainer] = getattr(self, "trainer", None)
        if trainer is None:
            return

        source = trainer._data_connector._train_dataloader_source
