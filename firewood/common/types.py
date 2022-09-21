from typing import Literal, Sequence, Union

import torch
from numpy import ndarray
from torch import Tensor

DEVICE = Union[str, torch.device]

STR = Union[str, Sequence[str]]
INT = Union[int, Sequence[int]]
FLOAT = Union[float, Sequence[float]]
NUMBER = Union[FLOAT, ndarray, Tensor]

NEST_STR = Union[str, Sequence[str], Sequence[Sequence[str]]]
NEST_INT = Union[int, Sequence[int], Sequence[Sequence[int]]]
NEST_FLOAT = Union[float, Sequence[float], Sequence[Sequence[float]]]

SAME_PADDING = Literal["same"]
