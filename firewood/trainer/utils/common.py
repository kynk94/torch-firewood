import os
from collections import defaultdict
from typing import Callable, Optional, Sequence, Union

from lightning_lite.utilities.rank_zero import _get_rank
from pytorch_lightning import Trainer
from torch import Tensor
from torch.optim import Optimizer

from firewood.utils.common import get_last_file, maximum_multiple_of_divisor


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
    if _get_rank() == 0:
        print(f"Found checkpoint: {checkpoint}")
    return checkpoint


def get_maximum_multiple_batch(input: Tensor, divisor: int) -> Tensor:
    """
    Get the batch that is the maximum multiple of the divisor.
    If divisor is larger than the batch size, return the `input`.
    """
    B = input.size(0)
    if B < divisor or B % divisor == 0:
        return input
    return input[: maximum_multiple_of_divisor(B, divisor)]


def reset_optimizers(
    obj: Union[Trainer, Optimizer, Sequence[Optimizer]]
) -> None:
    """
    Reset the optimizer state.

    Args:
        obj: Trainer or Optimizer or Sequence of Optimizers.
    """
    if isinstance(obj, Trainer):
        for optimizer in obj.optimizers:
            optimizer.state = defaultdict(dict)
    elif isinstance(obj, Optimizer):
        obj.state = defaultdict(dict)
    elif isinstance(obj, Sequence):
        for optimizer in obj:
            if not isinstance(optimizer, Optimizer):
                raise TypeError(f"{optimizer} is not an optimizer.")
            optimizer.state = defaultdict(dict)
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")
