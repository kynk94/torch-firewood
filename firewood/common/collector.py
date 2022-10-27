import inspect
import weakref
from typing import Callable, Dict, Iterator

import torch.nn as nn

from firewood import layers


class Collector:
    __layers: Dict[int, weakref.CallableProxyType] = dict()

    def __init__(self) -> None:
        raise RuntimeError("Collector is not meant to be instantiated.")

    @classmethod
    def add_layer(cls, layer: nn.Module) -> None:
        cls.__layers[id(layer)] = weakref.proxy(layer)

    @classmethod
    def del_layer(cls, layer: nn.Module) -> None:
        cls.__del_layer_by_id(id(layer))

    @classmethod
    def __del_layer_by_id(cls, id: int) -> None:
        cls.__layers.pop(id, None)

    @classmethod
    def layers(cls) -> Iterator[nn.Module]:
        for layer in cls.__layers.values():
            yield layer


def __layer_new_wrapper(cls) -> Callable[..., object]:
    func = cls.__new__

    def wrapper(*args, **kwargs) -> object:
        obj = func(args[0])
        Collector.add_layer(obj)
        return obj

    return wrapper


def __layer_del_wrapper(cls) -> Callable[..., None]:
    func = getattr(cls, "__del__", None)

    def wrapper(*args, **kwargs) -> None:
        self = args[0]
        Collector.del_layer(self)
        if func:
            func(self)

    return wrapper


def __assign_collector__() -> None:
    for name, cls in inspect.getmembers(layers):
        if inspect.isclass(cls) and issubclass(cls, nn.Module):
            cls.__new__ = __layer_new_wrapper(cls)
            cls.__del__ = __layer_del_wrapper(cls)


__assign_collector__()
