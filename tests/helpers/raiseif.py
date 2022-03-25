from types import TracebackType
from typing import (
    Any,
    Callable,
    Optional,
    Pattern,
    Tuple,
    Type,
    Union,
    overload,
)

import pytest
from _pytest._code import ExceptionInfo
from _pytest.python_api import E, RaisesContext


class _NormalContext:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        return exc_type is None


@overload
def raiseif(
    condition: bool,
    expected_exception: Union[Type[E], Tuple[Type[E], ...]],
    *,
    match: Optional[Union[str, Pattern[str]]] = ...,
) -> Union[_NormalContext, RaisesContext[E]]:
    ...


@overload
def raiseif(
    condition: bool,
    expected_exception: Union[Type[E], Tuple[Type[E], ...]],
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Union[_NormalContext, ExceptionInfo[E]]:
    ...


def raiseif(
    condition: bool,
    expected_exception: Union[Type[E], Tuple[Type[E], ...]],
    *args: Any,
    **kwargs: Any,
) -> Union[_NormalContext, RaisesContext[E], ExceptionInfo[E]]:
    if condition:
        return pytest.raises(expected_exception, *args, **kwargs)
    return _NormalContext()


def expect_raise(
    expected_exception: Union[Type[E], Tuple[Type[E], ...]],
    *args: Any,
    **kwargs: Any,
) -> Union[RaisesContext[E], ExceptionInfo[E]]:
    return pytest.raises(expected_exception, *args, **kwargs)
