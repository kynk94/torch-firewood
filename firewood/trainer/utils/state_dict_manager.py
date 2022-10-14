from collections.abc import MutableMapping
from typing import Any, Dict, Iterable

import torch
from torch import Tensor

from firewood.common.types import DEVICE
from firewood.utils import clone_to_cpu_tensor


class StateDictManager(MutableMapping):
    """
    Maintain a state_dict of parameters and their devices.
    If device is "cpu", can store the parameters on CPU and move them to the
    original device when they are accessed.
    """

    def __init__(self, device: DEVICE = "cpu", **kwargs: Tensor) -> None:
        self.device = torch.device(device)
        self._state_dict: Dict[str, Tensor] = dict()
        self._devices: Dict[str, torch.device] = dict()
        self.update(**kwargs)

    def __getitem__(self, key: str) -> Tensor:
        parameter = self._state_dict[key]
        device = self._devices[key]
        if self.device.type != "cpu":
            return parameter
        if device.type == "cpu":
            return parameter
        return parameter.to(device=device, non_blocking=True)

    def __setitem__(self, key: str, value: Tensor) -> None:
        if self.device.type == "cpu":
            store_value = clone_to_cpu_tensor(value)
        else:
            store_value = value
        self._state_dict[key] = store_value
        self._devices[key] = value.device

    def __delitem__(self, key: str) -> None:
        del self._state_dict[key]
        del self._devices[key]

    def __iter__(self) -> Iterable[str]:  # type: ignore
        return iter(self._state_dict)

    def __len__(self) -> int:
        return len(self._state_dict)

    def extract(self, key: str, pop_key: bool = True) -> Dict[str, Tensor]:
        return extract_state_dict(self._state_dict, key, pop_key)


def extract_state_dict(
    dictionary: Dict[str, Any], key: str, pop_key: bool = True
) -> Dict[str, Any]:
    extracted_state_dict = dict()
    if not key:
        raise ValueError("Argument `key` cannot be empty.")
    if not key.endswith("."):
        key += "."
    for k, v in dictionary.items():
        if not k.startswith(key):
            continue
        if pop_key:
            k = k[len(key) :]
        extracted_state_dict[k] = v
    return extracted_state_dict
