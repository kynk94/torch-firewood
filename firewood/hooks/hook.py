from typing import Any


def not_implemented(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


class _Hook:
    """
    Base class for all hooks.

    This class does not use `abc.ABC` to support flexible arguments.
    """

    apply = not_implemented
    remove = not_implemented
