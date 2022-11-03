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
    If gpu memory is not enough, set `store_on_cpu` as True. Then, the EMA
    weights stored in CPU memory.

    The shadow parameter is updated as:
        shadow_parameter = decay * shadow_parameter + (1 - decay) * parameter
    """

    def __init__(
        self,
        decay: float = 0.999,
        target_modules: Optional[STR] = None,
        exclude_modules: Optional[STR] = None,
        store_on_cpu: bool = False,
    ):
        super().__init__()
        self.decay = decay
        if isinstance(target_modules, str):
            target_modules = (target_modules,)
        elif isinstance(target_modules, Sequence):
            target_modules = tuple(target_modules)
        self.target_modules = target_modules
        if isinstance(exclude_modules, str):
            exclude_modules = (exclude_modules,)
        elif isinstance(exclude_modules, Sequence):
            exclude_modules = tuple(exclude_modules)
        self.exclude_modules = exclude_modules
        self.store_on_cpu = store_on_cpu

        self.shadow = StateDictManager(store_on_cpu=store_on_cpu)
        self.original = StateDictManager(store_on_cpu=True)
        self.device: Optional[DEVICE] = None

    @torch.no_grad()
    def _parameter_to_shadow(self, name: str, parameter: Tensor) -> None:
        if "inception" in name or "fid" in name or "lpips" in name:
            return
        if self.target_modules and not name.startswith(self.target_modules):
            return
        if self.exclude_modules and name.startswith(self.exclude_modules):
            return
        if name not in self.shadow:
            self.shadow[name] = parameter
            return
        self.shadow[name] = torch.lerp(parameter, self.shadow[name], self.decay)

    def on_train_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return
        if self.device is None:
            self.device = pl_module.device  # type: ignore
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
        if not trainer.is_global_zero:
            return
        for name, parameter in pl_module.named_parameters():
            self._parameter_to_shadow(name, parameter)

    def on_validation_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        self.original.update(pl_module.state_dict())
        self.shadow = trainer.strategy.broadcast(
            self.shadow.to("cpu"), src=0
        ).to(
            pl_module.device
        )  # type: ignore
        pl_module.load_state_dict(self.shadow, strict=False)
        self.shadow.clear()

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        if trainer.is_global_zero:
            self.shadow.update(pl_module.state_dict())
        pl_module.load_state_dict(self.original, strict=False)
        self.original.clear()

    def state_dict(self) -> Dict[str, Tensor]:
        return {k: args_to(v, device="cpu") for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for k, v in state_dict.items():
            self.shadow.update(k=args_to(v, device=self.device))
