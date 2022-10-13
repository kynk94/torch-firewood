import os
from typing import Callable, Optional, Union

from firewood.utils.common import get_last_file

from .data import update_train_batch_size_of_trainer
from .state_dict_manager import StateDictManager, extract_state_dict


def find_checkpoint(
    path: str,
    regex: str = "*.*pt*",  # .ckpt, .pt, .pth
    key: Optional[Union[Callable, str]] = None,
) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    if os.path.isdir(path):
        checkpoint = get_last_file(directory=path, regex=regex, key=key)
    else:
        checkpoint = path
    print(f"Found checkpoint: {checkpoint}")
    return checkpoint
