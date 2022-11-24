import inspect
import weakref
from typing import Any, Callable, Dict, Iterator, cast

import torch.nn as nn

from firewood import hooks, layers
from firewood.hooks.hook import _Hook


class Collector:
    __hooks: Dict[int, weakref.CallableProxyType] = dict()
    __layers: Dict[int, weakref.CallableProxyType] = dict()

    def __init__(self) -> None:
        raise RuntimeError("Collector is not meant to be instantiated.")

    @classmethod
    def add(cls, key: str, obj: Any) -> None:
        _id = id(obj)
        proxy = weakref.proxy(obj)
        if key == "hook":
            cls.__hooks[_id] = proxy
        elif key == "layer":
            cls.__layers[_id] = proxy
        else:
            raise ValueError(f"Unknown key: {key}")

    @classmethod
    def remove(cls, key: str, obj: Any) -> None:
        _id = id(obj)
        if key == "hook":
            cls.__hooks.pop(_id, None)
        elif key == "layer":
            cls.__layers.pop(_id, None)
        else:
            raise ValueError(f"Unknown key: {key}")

    @classmethod
    def hooks(cls) -> Iterator[Callable[..., Any]]:
        for hook in cls.__hooks.values():
            yield cast(Callable[..., Any], hook)

    @classmethod
    def layers(cls) -> Iterator[nn.Module]:
        for layer in cls.__layers.values():
            yield cast(nn.Module, layer)


def __new_wrapper(cls: Any, key: str) -> Callable[..., object]:
    func = cls.__new__

    def wrapper(*args: Any, **kwargs: Any) -> object:
        obj = func(args[0])
        Collector.add(key, obj)
        return obj

    return wrapper


def __del_wrapper(cls: Any, key: str) -> Callable[..., None]:
    func = getattr(cls, "__del__", None)

    def wrapper(*args: Any, **kwargs: Any) -> None:
        self = args[0]
        Collector.remove(key, self)
        if func:
            func(self)

    return wrapper


def __assign_collector() -> None:
    for name, cls in inspect.getmembers(hooks):
        if inspect.isclass(cls) and issubclass(cls, _Hook):
            setattr(cls, "__new__", __new_wrapper(cls, "hook"))
            setattr(cls, "__del__", __del_wrapper(cls, "hook"))
    for name, cls in inspect.getmembers(layers):
        if inspect.isclass(cls) and issubclass(cls, nn.Module):
            setattr(cls, "__new__", __new_wrapper(cls, "layer"))
            setattr(cls, "__del__", __del_wrapper(cls, "layer"))


__assign_collector()
