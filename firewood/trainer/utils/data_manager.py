from typing import Any, Mapping, Optional, Sequence

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.trainer.connectors.data_connector import (
    _DataLoaderSource,
)
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.data import (
    _get_dataloader_init_args_and_kwargs,
    _reinstantiate_wrapped_cls,
)
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader


def update_train_batch_size_of_trainer(
    trainer: Trainer, batch_size: int
) -> None:
    """Change batch size of train dataset."""

    def _get_batch_size_updated_dataloader(
        dataloader: Any, shuffle: bool = True
    ) -> DataLoader:
        sampler = trainer._data_connector._resolve_sampler(
            dataloader, shuffle=shuffle, mode=RunningStage.TRAINING
        )
        dl_args, dl_kwargs = _get_dataloader_init_args_and_kwargs(
            dataloader, sampler, mode=RunningStage.TRAINING
        )
        if "batch_size" in dl_kwargs:
            dl_kwargs.update(batch_size=batch_size)
        return _reinstantiate_wrapped_cls(dataloader, *dl_args, **dl_kwargs)

    source = trainer._data_connector._train_dataloader_source
    dataloader = source.dataloader()

    def train_dataloader() -> TRAIN_DATALOADERS:
        if isinstance(dataloader, Mapping):
            return {
                k: _get_batch_size_updated_dataloader(v)
                for k, v in dataloader.items()
            }
        if isinstance(dataloader, Sequence):
            return [_get_batch_size_updated_dataloader(v) for v in dataloader]
        return _get_batch_size_updated_dataloader(dataloader)

    datamodule: Optional[LightningDataModule] = getattr(
        trainer, "datamodule", None
    )
    if datamodule is None:
        trainer._data_connector._train_dataloader_source = _DataLoaderSource(
            train_dataloader(),
            source.name,
        )
    else:
        setattr(datamodule, "train_dataloader", train_dataloader)
