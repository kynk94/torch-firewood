import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, size: int, length: int) -> None:
        self.length = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index) -> Tensor:
        return self.data[index]

    def __len__(self):
        return self.length


class RandomImageDataset(Dataset):
    def __init__(self, size: int, length: int, channel: int = 3) -> None:
        self.length = length
        self.data = torch.randn(length, channel, size, size)

    def __getitem__(self, index) -> Tensor:
        return self.data[index]

    def __len__(self):
        return self.length


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def get_random_dataset(
    size: int,
    length: int,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return get_dataloader(
        RandomDataset(size, length), batch_size, shuffle, num_workers
    )


def get_random_image_dataset(
    size: int,
    length: int,
    channel: int = 3,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return get_dataloader(
        RandomImageDataset(size, length, channel),
        batch_size,
        shuffle,
        num_workers,
    )
