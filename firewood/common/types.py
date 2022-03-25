from typing import Literal, Sequence, Union

import torch

DEVICE = Union[str, torch.device]

STR = Union[str, Sequence[str]]
INT = Union[int, Sequence[int]]
FLOAT = Union[float, Sequence[float]]

NEST_STR = Union[str, Sequence[str], Sequence[Sequence[str]]]
NEST_INT = Union[int, Sequence[int], Sequence[Sequence[int]]]
NEST_FLOAT = Union[float, Sequence[float], Sequence[Sequence[float]]]

SAME_PADDING = Literal["same"]
