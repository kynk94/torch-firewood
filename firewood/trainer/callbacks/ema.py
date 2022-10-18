from typing import Any, Dict, Optional, Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from firewood.common.types import DEVICE, STR
from firewood.trainer.utils import StateDictManager
from firewood.utils import args_to


class ExponentialMovingAverage(Callback):
    """
    Maintains Exponential Moving Average of weights by decay factor.
    The EMA weights are updated after each batch and are used for validation.
    If gpu memory is not enough, set `device` as "cpu". Then, the EMA weights
    stored in CPU memory.

    The shadow parameter is updated as:
        shadow_parameter = decay * shadow_parameter + (1 - decay) * parameter
    """

    def __init__(
        self,
        decay: float = 0.999,
        target_modules: Optional[STR] = None,
        device: Optional[DEVICE] = None,
    ):
        super().__init__()
        self.decay = decay
        if isinstance(target_modules, str):
            target_modules = (target_modules,)
        elif isinstance(target_modules, Sequence):
            target_modules = tuple(target_modules)
        self.target_modules = target_modules
        self.device: Optional[DEVICE] = (
            torch.device(device) if device is not None else None
        )

        self.original = StateDictManager(device=self.device)
        self.shadow: Dict[str, Tensor] = dict()

    @torch.no_grad()
    def _parameter_to_shadow(self, name: str, parameter: Tensor) -> None:
        if isinstance(self.device, torch.device) and self.device.type == "cpu":
            parameter = parameter.detach().cpu()
        if name not in self.shadow:
            self.shadow[name] = parameter.clone()
            return
        self.shadow[name].copy_(
            torch.lerp(parameter, self.shadow[name], self.decay)
        )

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.global_rank != 0:
            return
        if self.device is None:
            self.device = self.original.device = torch.device(pl_module.device)
        for name, parameter in pl_module.named_parameters():
            self._parameter_to_shadow(name, parameter)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_rank != 0:
            return
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
        if trainer.sanity_checking:
            return

        self.original.update(pl_module.state_dict())
        self.shadow = trainer.strategy.broadcast(self.shadow, src=0)
        pl_module.load_state_dict(self.shadow, strict=False)
        if trainer.global_rank > 0:
            self.shadow.clear()

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        pl_module.load_state_dict(self.original, strict=False)
        self.original.clear()

    def state_dict(self) -> Dict[str, Tensor]:
        return {k: args_to(v, device="cpu") for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.shadow = {
            k: args_to(v, device=self.device) for k, v in state_dict.items()
        }
