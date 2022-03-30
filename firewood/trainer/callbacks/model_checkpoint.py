import time
from datetime import timedelta
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint
from pytorch_lightning.utilities.types import _PATH, STEP_OUTPUT

_MINUTE_LOG_NAME = "epoch/elapsed_minute"


class ModelCheckpoint(_ModelCheckpoint):
    def __init__(
        self,
        dirpath: Optional[_PATH] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_last_k: Optional[int] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
    ):
        if save_last_k is not None:
            if monitor is not None:
                raise ValueError(
                    "save_last_k is only supported when monitor is not set"
                )
            monitor = _MINUTE_LOG_NAME
            mode = "max"
            save_top_k = save_last_k
            save_on_train_epoch_end = True
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )
        self.initial_time = time.monotonic()

    def on_train_batch_end(  # type: ignore[override]
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        interval_minutes = (time.monotonic() - self.initial_time) / 60
        pl_module.log(_MINUTE_LOG_NAME, interval_minutes, on_step=True)
        super().on_train_batch_end(
            trainer=trainer,
            pl_module=pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=batch_idx,
        )
