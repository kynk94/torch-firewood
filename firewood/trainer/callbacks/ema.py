from typing import Any, Dict, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from firewood.common.types import STR
from firewood.trainer.utils import StateDictManager
from firewood.utils import clone_to_cpu_tensor


class ExponentialMovingAverage(Callback):
    """
    Maintains Exponential Moving Average of weights by decay factor.
    The EMA weights are updated after each batch and are used for validation.
    And stored in CPU memory.

    The shadow parameter is updated as:
        shadow_parameter = decay * shadow_parameter + (1 - decay) * parameter
    """

    def __init__(
        self, decay: float = 0.999, target_modules: Optional[STR] = None
    ):
        super().__init__()
        self.decay = decay
        if isinstance(target_modules, str):
            target_modules = (target_modules,)
        self.target_modules = target_modules

        self.original = StateDictManager()
        self.shadow: Dict[str, Tensor] = dict()

    @torch.no_grad()
    def _parameter_to_shadow(self, name: str, parameter: Tensor) -> None:
        parameter = clone_to_cpu_tensor(parameter)
        if name not in self.shadow:
            self.shadow[name] = parameter
            return
        self.shadow[name] = (
            self.decay * self.shadow[name] + (1 - self.decay) * parameter
        )

    @rank_zero_only
    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        for name, param in pl_module.named_parameters():
            self._parameter_to_shadow(name, param)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.target_modules is None:
            for name, parameter in pl_module.named_parameters():
                self._parameter_to_shadow(name, parameter)
        else:
            for name, parameter in pl_module.named_parameters():
                if name.startswith(self.target_modules):
                    self._parameter_to_shadow(name, parameter)

    def on_validation_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self.shadow:
            return

        self.original.update(pl_module.state_dict())
        pl_module.load_state_dict(self.shadow, strict=False)

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not self.shadow:
            return

        pl_module.load_state_dict(self.original, strict=False)

    def state_dict(self) -> None:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.shadow = state_dict
