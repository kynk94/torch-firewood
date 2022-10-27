import functools
import glob
import inspect
import os
import random
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import torch
import torchvision.datasets as TD
from natsort import natsort_keygen
from numpy import ndarray
from PIL import Image
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.trainer.connectors.data_connector import (
    _DataLoaderSource,
)
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.data import (
    _get_dataloader_init_args_and_kwargs,
    _reinstantiate_wrapped_cls,
)
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder, VisionDataset

from firewood.common.types import INT
from firewood.utils.common import normalize_int_tuple

try:
    import albumentations as A
except (AttributeError, ImportError, ModuleNotFoundError):
    A = None


Image.init()

DATASET = Union[Dataset, Tuple[Dataset, ...], List[Dataset]]
DATALOADER = Union[DataLoader, Tuple[DataLoader, ...], List[DataLoader]]
SPLIT = Union[
    Literal["auto", "train/test/val", "train/test", "train/val"],
    Tuple[float, ...],
    List[float],
]
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)
MODE = Literal[
    "1",
    "CMYK",
    "F",
    "HSV",
    "I",
    "L",
    "LAB",
    "P",
    "RGB",
    "RGBA",
    "RGBX",
    "YCbCr",
    "LA",
]  # one of Image.MODES


def image_transform(
    image: Any,
    transform: Optional[Callable[..., Any]] = None,
    is_albumentation: bool = False,
) -> Any:
    if transform is None:
        return image
    if is_albumentation:
        if not isinstance(image, ndarray):
            image = np.array(image)
        return transform(image=image)["image"]
    return transform(image)


def multi_image_transform(
    images: Any,
    transform: Optional[Callable[..., Any]] = None,
    is_albumentation: bool = False,
) -> Tuple[Any, ...]:
    if transform is None:
        return images
    if not is_albumentation:
        return tuple(transform(image) for image in images)

    images = tuple(
        np.array(image) if isinstance(image, Image.Image) else image
        for image in images
    )
    if not isinstance(transform, A.ReplayCompose):
        transform = A.ReplayCompose([transform])
    data = transform(image=images[0])  # type: ignore
    outputs = [data["image"]]
    for i in range(1, len(images)):
        output = A.ReplayCompose.replay(data["replay"], image=images[i])
        outputs.append(output["image"])
    return tuple(outputs)


def image_masks_transform(
    image: Any,
    masks: Any,
    transform: Optional[Callable[..., Any]] = None,
    is_albumentation: bool = False,
) -> Tuple[Any, Any]:
    if transform is None:
        return image, masks
    if not is_albumentation:
        if isinstance(masks, Image.Image):
            return transform(image), transform(masks)
        return transform(image), multi_image_transform(masks, transform, False)

    if not isinstance(image, ndarray):
        image = np.array(image)
    if isinstance(masks, Image.Image):
        masks = np.array(masks)
        data = transform(image=image, mask=masks)
        return data["image"], data["mask"]
    masks = tuple(
        np.array(mask) if not isinstance(mask, ndarray) else mask
        for mask in masks
    )
    data = transform(image=image, masks=masks)
    return data["image"], data["masks"]


def multi_image_masks_transform(
    images: Any,
    masks: Any,
    transform: Optional[Callable[..., Any]] = None,
    is_albumentation: bool = False,
) -> Tuple[Any, ...]:
    if transform is None:
        return images, masks
    if not is_albumentation:
        images = multi_image_transform(images, transform, False)
        masks = multi_image_transform(masks, transform, False)
        return images, masks

    images = tuple(
        np.array(image) if isinstance(image, Image.Image) else image
        for image in images
    )
    if not isinstance(transform, A.ReplayCompose):
        transform = A.ReplayCompose([transform])

    if isinstance(masks, Image.Image):
        masks = np.array(masks)
        data = transform(image=images[0], mask=masks)  # type: ignore
        outputs = [data["image"]]
        for i in range(1, len(images)):
            output = A.ReplayCompose.replay(data["replay"], image=images[i])
            outputs.append(output["image"])
        return tuple(outputs), data["mask"]

    masks = tuple(
        np.array(mask) if not isinstance(mask, ndarray) else mask
        for mask in masks
    )
    data = transform(image=images[0], masks=masks)  # type: ignore
    outputs = [data["image"]]
    for i in range(1, len(images)):
        output = A.ReplayCompose.replay(data["replay"], image=images[i])
        outputs.append(output["image"])
    return tuple(outputs), data["masks"]


class PillowLoader:
    def __init__(self, mode: MODE = "RGB"):
        self.mode = mode

    def __call__(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            return Image.open(f).convert(mode=self.mode)


def make_condition_dataset(
    image_directory: str,
    condition_directory: str,
    extensions: Optional[Union[Tuple[str, ...], Set[str], List[str]]] = None,
    condition_extension: str = ".npy",
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, str]]:
    image_directory = os.path.expanduser(image_directory)
    condition_directory = os.path.expanduser(condition_directory)

    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Only one of extensions and is_valid_file should be specified"
        )
    if extensions is not None:
        tuple_extensions = tuple(extensions)

        def is_valid_file(x: str) -> bool:
            return TD.folder.has_file_allowed_extension(x, tuple_extensions)

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    for file in glob.glob(
        os.path.join(image_directory, "**", "*.*"), recursive=True
    ):
        if not is_valid_file(file):
            continue
        basename = os.path.basename(file)
        name, _ = os.path.splitext(basename)
        condition_basename = name + condition_extension
        relpath = os.path.relpath(file, image_directory)
        if basename == relpath:
            condition_file = os.path.join(
                condition_directory, condition_basename
            )
        else:
            rel_directory = os.path.dirname(relpath)
            condition_file = os.path.join(
                condition_directory, rel_directory, condition_basename
            )
        item = (file, condition_file)
        instances.append(item)
    instances.sort(key=natsort_keygen())
    return instances


def make_no_class_dataset(
    directory: str,
    extensions: Optional[Union[Tuple[str, ...], Set[str], List[str]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    directory = os.path.expanduser(directory)

    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Only one of extensions and is_valid_file should be specified"
        )
    if extensions is not None:
        tuple_extensions = tuple(extensions)

        def is_valid_file(x: str) -> bool:
            return TD.folder.has_file_allowed_extension(x, tuple_extensions)

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    for file in glob.glob(os.path.join(directory, "**", "*.*"), recursive=True):
        if not is_valid_file(file):
            continue
        item = (file, 0)
        instances.append(item)
    instances.sort(key=natsort_keygen())
    return instances


class ConditionDatasetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        condition_root: str,
        loader: Callable[[str], Any],
        use_albumentations: bool = False,
        extensions: Optional[Tuple[str, ...]] = None,
        condition_extension: str = ".npy",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=os.path.normpath(root),
            transform=transform,
            target_transform=target_transform,
        )
        self.condition_root = os.path.normpath(condition_root)
        self.loader = loader
        self.use_albumentations = use_albumentations
        self.extensions = extensions
        self.samples = make_condition_dataset(
            image_directory=self.root,
            condition_directory=self.condition_root,
            extensions=extensions,
            condition_extension=condition_extension,
            is_valid_file=is_valid_file,
        )
        self.targets = [sample[-1] for sample in self.samples]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image_path, condition_path = self.samples[index]
        image = self.loader(image_path)
        condition = np.load(condition_path)
        image = image_transform(image, self.transform, self.use_albumentations)
        if self.target_transform is not None:
            condition = self.target_transform(condition)
        return image, condition

    def __len__(self) -> int:
        return len(self.samples)


class ConditionImageFolder(ConditionDatasetFolder):
    def __init__(
        self,
        root: str,
        condition_root: str,
        condition_extension: str = ".npy",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_albumentations: bool = False,
        loader: Optional[Callable[[str], Any]] = None,
        loader_mode: MODE = "RGB",
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if loader is not None and loader_mode != "RGB":
            raise ValueError(
                "loader_mode is not supported when loader is not None"
            )
        super().__init__(
            root=root,
            condition_root=condition_root,
            loader=loader or PillowLoader(mode=loader_mode),
            use_albumentations=use_albumentations,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            condition_extension=condition_extension,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )


class NoClassDatasetFolder(VisionDataset):
    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        use_albumentations: bool = False,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=os.path.normpath(root),
            transform=transform,
            target_transform=target_transform,
        )
        self.loader = loader
        self.use_albumentations = use_albumentations
        self.extensions = extensions
        self.samples = make_no_class_dataset(
            directory=self.root,
            extensions=extensions,
            is_valid_file=is_valid_file,
        )
        self.targets = [sample[-1] for sample in self.samples]  # all zeros

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = image_transform(
            sample, self.transform, self.use_albumentations
        )
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class NoClassImageFolder(NoClassDatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_albumentations: bool = False,
        loader: Optional[Callable[[str], Any]] = None,
        loader_mode: MODE = "RGB",
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        if loader is not None and loader_mode != "RGB":
            raise ValueError(
                "loader_mode is not supported when loader is not None"
            )
        super().__init__(
            root=root,
            loader=loader or PillowLoader(mode=loader_mode),
            use_albumentations=use_albumentations,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )


class PairedImageFolder(NoClassImageFolder):
    """
    Dataset for Image to Image Translation tasks.
    A generic data loader where the paired images are arranged in this way: ::

        root/xxx.png
        root/xxy.png
        root/something/xxz.png

    Each image is a merged image along the horizontal or vertical axes
    of the source and target.
    Since each corresponds to each target, there is no need class dir.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        source_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_albumentations: bool = False,
        target_type: str = "image",  # "image" or "mask"
        loader: Optional[Callable[[str], Any]] = None,
        loader_mode: MODE = "RGB",
        is_valid_file: Optional[Callable[[str], bool]] = None,
        axis: Union[str, int] = "horizontal",
    ) -> None:
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            use_albumentations=use_albumentations,
            loader=loader,
            loader_mode=loader_mode,
            is_valid_file=is_valid_file,
        )
        self.source_transform = source_transform
        self.target_type = target_type
        if self.use_albumentations and self.target_type == "image":
            self.transform = A.ReplayCompose(self.transform)  # type: ignore

        if isinstance(axis, str):
            if axis.lower().startswith("v"):
                self.axis = 0
            elif axis.lower().startswith("h"):
                self.axis = 1
            else:
                raise ValueError(
                    f"axis must be either 'horizontal' or 'vertical'"
                )
        elif axis in {0, 1}:
            self.axis = axis
        else:
            raise ValueError(f"axis must be either 0 or 1, got {axis}")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, _ = self.samples[index]
        sample: Image.Image = self.loader(path)
        if not isinstance(sample, Image.Image):
            sample = Image.fromarray(sample)
        width, height = sample.size
        if self.axis == 0:
            split_point = height // 2
            source = sample.crop((0, 0, width, split_point))
            target = sample.crop((0, split_point, width, height))
        else:
            split_point = width // 2
            source = sample.crop((0, 0, split_point, height))
            target = sample.crop((split_point, 0, width, height))

        source, target = multi_image_transform(
            (source, target), self.transform, self.use_albumentations
        )
        source = image_transform(
            source, self.source_transform, self.use_albumentations
        )
        target = image_transform(
            target, self.target_transform, self.use_albumentations
        )
        return source, target


def _get_train_val_test_dir(
    path: str,
) -> Tuple[str, Optional[str], Optional[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")

    train_dir = os.path.join(path, "train")
    if not os.path.exists(train_dir):
        return path, None, None

    if os.path.exists(os.path.join(path, "val")):
        val_dir = os.path.join(path, "val")
    else:
        val_dir = None

    if os.path.exists(os.path.join(path, "test")):
        test_dir = os.path.join(path, "test")
    else:
        test_dir = None
    return train_dir, val_dir, test_dir


def get_train_val_test_datasets(
    root: str,
    dataset_class: VisionDataset = ImageFolder,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    loader: Optional[Callable[[str], Any]] = None,
    loader_mode: MODE = "RGB",
    is_valid_file: Optional[Callable[[str], bool]] = None,
    split: Optional[SPLIT] = None,
    **kwargs: Any,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    train_dir, val_dir, test_dir = _get_train_val_test_dir(root)

    if loader is not None and loader_mode != "RGB":
        raise ValueError("loader_mode is not supported when loader is not None")
    loader = loader or PillowLoader(mode=loader_mode)

    train_dataset = dataset_class(
        root=train_dir,
        transform=transform,
        target_transform=target_transform,
        loader=loader,
        is_valid_file=is_valid_file,
        **kwargs,
    )
    if val_dir:
        val_dataset = dataset_class(
            root=val_dir,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
            **kwargs,
        )
    else:
        val_dataset = None
    if test_dir:
        test_dataset = dataset_class(
            root=test_dir,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
            **kwargs,
        )
    else:
        test_dataset = None

    datasets = (train_dataset, val_dataset, test_dataset)
    if split is None:
        return datasets
    if isinstance(split, str):
        split = split.strip().lower()  # type: ignore
        if (
            (split == "auto" or ("val" in split and "test" in split))
            and val_dataset is not None
            and test_dataset is not None
        ):
            return datasets
        if "val" in split and val_dataset is not None:
            if test_dataset is not None:
                val_dataset = ConcatDataset([val_dataset, test_dataset])
            return train_dataset, val_dataset, None
        if "test" in split and test_dataset is not None:
            if val_dataset is not None:
                test_dataset = ConcatDataset([val_dataset, test_dataset])
            return train_dataset, None, test_dataset
    return concat_and_split_datasets(
        datasets=datasets, split=split, shuffle=False
    )


def concat_and_split_datasets(
    datasets: Union[DATASET, Mapping[Any, Dataset]],
    split: SPLIT = "auto",
    shuffle: bool = False,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """
    Concatenate datasets and split them into train, val, and test datasets.

    Args:
        datasets: A tuple of datasets to concatenate.
        split: A tuple of floats or a string. If a tuple of floats, the sum of
            the floats must be 1.0. If a string, it must be one of the following
            values: "auto" or permutation of "train", "val", and "test".
            If "auto", the split will be (0.8, 0.1, 0.1).
        shuffle: Whether to shuffle the dataset before splitting.
    """
    # Concatenate datasets
    if isinstance(datasets, Mapping):
        datasets = tuple(datasets.values())
    if isinstance(datasets, (list, tuple)):
        datasets = tuple(dataset for dataset in datasets if dataset is not None)
        if len(datasets) == 0:
            raise ValueError("`datasets` must not be empty")
        elif len(datasets) == 1:
            datasets = datasets[0]
        else:
            datasets = ConcatDataset(datasets)

    # Split concatenated dataset to train, val, test datasets
    if isinstance(split, str):
        split = split.strip().lower()  # type: ignore
        if split == "auto" or ("val" in split and "test" in split):
            split = (0.8, 0.1, 0.1)
        elif "val" in split:
            split = (0.9, 0.1, 0.0)
        elif "test" in split:
            split = (0.9, 0.0, 0.1)
        else:
            raise ValueError(f"Not supported split: {split}")
    split = tuple(0 if rate is None or rate < 0 else rate for rate in split)
    if len(split) not in {2, 3}:
        raise ValueError("`split` must be a tuple which length is 2 or 3")
    if sum(split) != 1 or split[0] == 0 or any(rate == 1 for rate in split):
        raise ValueError(
            "`split` must be a tuple of float numbers, "
            "each of which is between 0 and 1, "
            "and the sum of all numbers must be 1"
        )

    indices: List[Optional[Sequence[int]]] = []
    start = 0
    for i in range(3):
        end = round(sum(split[: i + 1]) * len(datasets))  # type: ignore
        if start == end:
            indices.append(None)
        else:
            indices.append(range(start, end))
        start = end

    if shuffle:
        index_list = list(range(len(datasets)))  # type: ignore
        random.shuffle(index_list)
        indices = [
            index_list[i.start : i.stop] if isinstance(i, range) else None
            for i in indices
        ]

    train_dataset = Subset(datasets, cast(Sequence[int], indices[0]))
    if indices[1] is not None:
        val_dataset = Subset(datasets, indices[1])
    else:
        val_dataset = None
    if indices[2] is not None:
        test_dataset = Subset(datasets, indices[2])
    else:
        test_dataset = None
    return train_dataset, val_dataset, test_dataset


def torchvision_train_val_test_datasets(
    name: str,
    root: str = "./datasets",
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    **kwargs: Any,
) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    original_kwargs = kwargs.copy()
    lower_name = name.lower()
    root = os.path.normpath(root)

    all_datasets = tuple(dataset.lower() for dataset in TD.__all__)
    if lower_name not in all_datasets:
        raise ValueError(f"{name} is not a valid torchvision dataset")
    found_dataset_name = TD.__all__[all_datasets.index(lower_name)]
    found_dataset = getattr(TD, found_dataset_name)
    arg_spec = inspect.getfullargspec(found_dataset.__init__).args

    # for all datasets
    if "root" in arg_spec:
        kwargs["root"] = root
    if "download" in arg_spec:
        kwargs["download"] = True
    if "transform" in arg_spec:
        kwargs["transform"] = transform
    if "target_transform" in arg_spec:
        kwargs["target_transform"] = target_transform

    # train datasets
    if "train" in arg_spec:
        kwargs["train"] = True
    elif "split" in arg_spec:
        kwargs["split"] = "train"
    elif "classes" in arg_spec:
        if original_kwargs.get("classes") is not None:
            kwargs["classes"] = original_kwargs["classes"] + "_train"
        else:
            kwargs["classes"] = "train"
    try:
        train_dataset = found_dataset(**kwargs)
    except RuntimeError:
        if "download" in arg_spec:
            kwargs["download"] = None
        train_dataset = found_dataset(**kwargs)

    # val datasets
    if "split" in arg_spec:
        kwargs["split"] = "val"
        try:
            val_dataset = found_dataset(**kwargs)
        except ValueError:
            val_dataset = None
    elif "classes" in arg_spec:
        if original_kwargs.get("classes") is not None:
            kwargs["classes"] = original_kwargs["classes"] + "_val"
        else:
            kwargs["classes"] = "val"
        try:
            val_dataset = found_dataset(**kwargs)
        except ValueError:
            val_dataset = None
    else:
        val_dataset = None

    # test datasets
    if "train" in arg_spec:
        kwargs["train"] = False
        test_dataset = found_dataset(**kwargs)
    elif "split" in arg_spec:
        kwargs["split"] = "test"
        try:
            test_dataset = found_dataset(**kwargs)
        except ValueError:
            test_dataset = None
    elif "classes" in arg_spec:
        if original_kwargs.get("classes") is not None:
            kwargs["classes"] = original_kwargs["classes"] + "_test"
        else:
            kwargs["classes"] = "test"
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset


def _check_datasets(
    *__datasets: Dataset,
    datasets: Optional[DATASET] = None,
) -> Tuple[Dataset, ...]:
    if datasets is None:
        if not __datasets:
            raise ValueError("No datasets provided")
        if len(__datasets) == 1 and isinstance(__datasets[0], (tuple, list)):
            datasets = cast(Tuple[Dataset, ...], __datasets[0])
        else:
            datasets = __datasets
    elif isinstance(datasets, Dataset):
        datasets = (datasets,)
    if all(dataset is None for dataset in datasets):
        raise ValueError("No datasets provided")
    return tuple(datasets)


def get_dataloaders(
    *__datasets: Dataset,
    datasets: Optional[DATASET] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    shuffle_others: bool = False,
    num_workers: int = 4,
    pin_memory: bool = False,
    **kwargs: Any,
) -> Tuple[DataLoader, ...]:
    dataloaders: List[DataLoader] = []
    for dataset in _check_datasets(*__datasets, datasets=datasets):
        if dataset is None:
            dataloaders.append(None)
            continue
        dataloaders.append(
            DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                **kwargs,
            )
        )
        if not shuffle_others:
            shuffle = False
    return tuple(dataloaders)


def get_lightning_datamodule(
    *__datasets: Dataset,
    datasets: Optional[DATASET] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> LightningDataModule:
    datasets = _check_datasets(*__datasets, datasets=datasets)
    if len(datasets) == 1:
        train_dataset = datasets[0]
        val_dataset = None
        test_dataset = None
    elif len(datasets) == 2:
        train_dataset = datasets[0]
        val_dataset = datasets[1]
        test_dataset = None
    elif len(datasets) == 3:
        train_dataset = datasets[0]
        val_dataset = datasets[1]
        test_dataset = datasets[2]
    else:
        raise ValueError("Too many datasets provided")

    return LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        predict_dataset=None,
        num_workers=num_workers,
    )


class DataModule(LightningDataModule):
    train_dataset: Dataset
    val_dataset: Optional[Dataset]
    test_dataset: Optional[Dataset]
    predict_dataset: Optional[Dataset]

    def __init__(
        self,
        *__datasets: Dataset,
        datasets: Optional[DATASET] = None,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        if not hasattr(self, "from_datasets"):
            raise ImportError(
                "LightningDataModule is not available. "
                "Please install PyTorch Lightning by following command: "
                "pip install pytorch-lightning"
            )
        self.__attach_datasets(_check_datasets(*__datasets, datasets=datasets))

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def __attach_datasets(self, datasets: Tuple[Dataset, ...]) -> None:
        if len(datasets) == 1:
            self.train_dataset = datasets[0]
            self.val_dataset = None
            self.test_dataset = None
            self.predict_dataset = None
        elif len(datasets) == 2:
            self.train_dataset = datasets[0]
            self.val_dataset = datasets[1]
            self.test_dataset = None
            self.predict_dataset = None
        elif len(datasets) == 3:
            self.train_dataset = datasets[0]
            self.val_dataset = datasets[1]
            self.test_dataset = datasets[2]
            self.predict_dataset = None
        elif len(datasets) == 4:
            self.train_dataset = datasets[0]
            self.val_dataset = datasets[1]
            self.test_dataset = datasets[2]
            self.predict_dataset = datasets[3]
        else:
            raise ValueError("Too many datasets provided")

        if self.val_dataset is None:
            setattr(self, "val_dataloader", super().val_dataloader)
        if self.test_dataset is None:
            setattr(self, "test_dataloader", super().test_dataloader)
        if self.predict_dataset is None:
            setattr(self, "predict_dataloader", super().predict_dataloader)

    @overload
    def __dataloader(
        self,
        datasets: Dataset,
        shuffle: bool,
    ) -> DataLoader:
        ...

    @overload
    def __dataloader(
        self,
        datasets: Union[Tuple[Dataset, ...], List[Dataset]],
        shuffle: bool,
    ) -> Tuple[DataLoader, ...]:
        ...

    def __dataloader(
        self,
        datasets: DATASET,
        shuffle: bool = False,
    ) -> DATALOADER:
        if isinstance(datasets, (tuple, list)):
            return tuple(
                self.__dataloader(dataset, shuffle) for dataset in datasets
            )
        return self.__single_dataloader(datasets, shuffle)

    def __single_dataloader(
        self, dataset: Dataset, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DATALOADER:
        return self.__dataloader(self.train_dataset, True)

    def val_dataloader(self) -> DATALOADER:
        return self.__dataloader(cast(Dataset, self.val_dataset), False)

    def test_dataloader(self) -> DATALOADER:
        return self.__dataloader(cast(Dataset, self.test_dataset), False)

    def predict_dataloader(self) -> DATALOADER:
        return self.__dataloader(cast(Dataset, self.predict_dataset), False)


def update_resize_of_vision_dataset(
    dataset: VisionDataset, resolution: INT
) -> Dataset:
    if isinstance(dataset, ConcatDataset):
        for _dataset in dataset.datasets:
            update_resize_of_vision_dataset(_dataset, resolution)
        return dataset
    if isinstance(dataset, Subset):
        update_resize_of_vision_dataset(dataset.dataset, resolution)
        return dataset

    if getattr(dataset, "transform", None) is None:
        raise ValueError("Dataset does not have transform attribute.")

    resolution = normalize_int_tuple(resolution, 2)
    if isinstance(dataset.transform, (transforms.Resize, A.Resize)):
        transform = [dataset.transform]
    elif isinstance(dataset.transform, (transforms.Compose, A.Compose)):
        transform = list(dataset.transform.transforms)
    for t in transform:
        if isinstance(t, transforms.Resize):
            t.size = resolution
            return dataset
        if isinstance(t, A.Resize):
            t.height, t.width = resolution
            return dataset
    raise ValueError("Dataset does not have Resize transform.")


def __get_batch_size_updated_dataloader(
    trainer: Trainer,
    dataloader: Any,
    batch_size: int,
    shuffle: bool = True,
    mode: RunningStage = RunningStage.TRAINING,
) -> DataLoader:
    sampler = trainer._data_connector._resolve_sampler(
        dataloader, shuffle=shuffle, mode=mode
    )
    if hasattr(sampler, "batch_size"):
        setattr(sampler, "batch_size", batch_size)
    dl_args, dl_kwargs = _get_dataloader_init_args_and_kwargs(
        dataloader, sampler, mode=mode
    )
    if len(dl_args) > 2 and isinstance(dl_args[1], int):  # type: ignore
        dl_args = cast(Tuple[Any], (dl_args[0], batch_size, *dl_args[2:]))
    else:
        dl_kwargs.update(batch_size=batch_size)
    return _reinstantiate_wrapped_cls(dataloader, *dl_args, **dl_kwargs)


def __updated_dataloader(
    trainer: Trainer,
    dataloader: Any,
    batch_size: int,
    shuffle: bool,
    mode: RunningStage,
) -> TRAIN_DATALOADERS:
    if batch_size is None:
        return dataloader
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, mode=mode)
    if isinstance(dataloader, Mapping):
        return {
            k: __get_batch_size_updated_dataloader(trainer, v, **kwargs)  # type: ignore
            for k, v in dataloader.items()
        }
    if isinstance(dataloader, Sequence):
        return [
            __get_batch_size_updated_dataloader(trainer, v, **kwargs)  # type: ignore
            for v in dataloader
        ]
    return __get_batch_size_updated_dataloader(trainer, dataloader, **kwargs)  # type: ignore


def update_dataloader_of_trainer(
    trainer: Trainer,
    target: str = "train/val",
    batch_size: Optional[int] = None,
    resolution: Optional[INT] = None,
    mode: RunningStage = RunningStage.TRAINING,
) -> None:
    """
    Update batch size and resolution of dataloader of trainer.
    If get invalid CUDA memory access error, it means CUDA memory has overflowed.
    """
    if batch_size is None and resolution is None:
        return

    targets = []
    if "train" in target:
        targets.append("train")
    if "val" in target:
        targets.append("val")
    if "test" in target:
        targets.append("test")
    if "predict" in target or not targets:
        raise ValueError(f"Invalid target: {target}")

    torch.cuda.empty_cache()

    for target in targets:
        source: _DataLoaderSource = getattr(
            trainer._data_connector, f"_{target}_dataloader_source"
        )
        dataloader = source.dataloader()
        if resolution is not None:
            resolution = normalize_int_tuple(resolution, 2)
            if isinstance(dataloader, Mapping):
                for v in dataloader.values():
                    dataset = cast(DataLoader, v).dataset
                    update_resize_of_vision_dataset(dataset, resolution)
            elif isinstance(dataloader, Sequence):
                for v in dataloader:
                    dataset = cast(DataLoader, v).dataset
                    update_resize_of_vision_dataset(dataset, resolution)
            else:
                update_resize_of_vision_dataset(dataloader.dataset, resolution)

        datamodule: Optional[LightningDataModule] = getattr(
            trainer, "datamodule", None
        )
        kwargs = dict(
            trainer=trainer,
            dataloader=dataloader,
            batch_size=batch_size,
            shuffle=target == "train",
            mode=mode,
        )
        if datamodule is None:
            setattr(
                trainer._data_connector,
                f"_{target}_dataloader_source",
                _DataLoaderSource(__updated_dataloader(**kwargs), source.name),  # type: ignore
            )
        else:
            setattr(
                datamodule,
                f"{target}_dataloader",
                functools.partial(__updated_dataloader, **kwargs),
            )

        getattr(trainer, f"reset_{target}_dataloader")()
        if target == "train":
            data_fetcher = trainer.fit_loop._data_fetcher
            _dataloader = trainer.train_dataloader
        elif target == "val":
            data_fetcher = trainer.fit_loop.epoch_loop.val_loop._data_fetcher
            _dataloader = trainer.val_dataloaders
        elif target == "test":
            data_fetcher = trainer.test_loop._data_fetcher
            _dataloader = trainer.test_dataloaders
        else:
            raise ValueError("Invalid target.")
        if data_fetcher is None:
            continue
        if isinstance(_dataloader, list) and len(_dataloader) == 1:
            _dataloader = _dataloader[0]
        data_fetcher.setup(
            cast(DataLoader, _dataloader),
            batch_to_device=getattr(data_fetcher, "batch_to_device", None),
        )
        data_fetcher.dataloader_iter = iter(data_fetcher.dataloader)
        data_fetcher.reset()

    progress_bar = trainer.progress_bar_callback
    if isinstance(progress_bar, TQDMProgressBar):
        progress_bar.main_progress_bar.total = (
            progress_bar.total_batches_current_epoch
        )
        progress_bar.main_progress_bar.refresh()
