import glob
import os
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

from numpy import ndarray
from packaging.version import Version
from pkg_resources import get_distribution
from torch import Tensor

from firewood.common.types import FLOAT, INT, NUMBER, STR


@overload
def clamp(input: float, _min: float, _max: float) -> float:
    ...


@overload
def clamp(input: Sequence[float], _min: float, _max: float) -> Sequence[float]:
    ...


@overload
def clamp(
    input: ndarray, _min: Union[float, ndarray], _max: Union[float, ndarray]
) -> ndarray:
    ...


@overload
def clamp(
    input: Tensor, _min: Union[float, Tensor], _max: Union[float, Tensor]
) -> Tensor:
    ...


def clamp(input: NUMBER, _min: Any, _max: Any) -> NUMBER:
    if isinstance(input, float):
        return max(_min, min(input, _max))
    if isinstance(input, Sequence):
        return type(input)(clamp(i, _min, _max) for i in input)  # type: ignore
    if isinstance(input, ndarray):
        return input.clip(_min, _max)
    return input.clamp(_min, _max)


def is_newer_torch(version: Union[str, int, float]) -> bool:
    if isinstance(version, (int, float)):
        version = str(version)
    torch_version = Version(get_distribution("torch").version)
    return torch_version >= Version(version)


def is_older_torch(version: Union[str, int, float]) -> bool:
    return not is_newer_torch(version)


def get_nested_first(obj: Any) -> Any:
    if isinstance(obj, Sequence) and obj:
        return get_nested_first(next(iter(obj)))
    return obj


def get_nested_last(obj: Any) -> Any:
    if isinstance(obj, Sequence) and obj:
        return get_nested_last(next(reversed(obj)))
    return obj


def get_name(object: object) -> str:
    name = getattr(object, "__name__", object.__class__.__name__)
    return name


def validate_filename(filename: str) -> str:
    filename = str(filename).strip().replace(" ", "-")
    return re.sub(r"_+", "_", re.sub(r"(?u)[^-\w.]", "_", filename))


def updated_dict(
    dictionary: Dict[str, Any],
    *args: Dict[str, Any],
    delete: Optional[STR] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    copy = dictionary.copy()
    if args:
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError("Argument must be a dictionary.")
            copy.update(arg)
    if kwargs:
        copy.update(kwargs)
    if delete:
        if isinstance(delete, str):
            delete = (delete,)
        for key in delete:
            if key in copy:
                del copy[key]
    return copy


def search_kwargs(
    kwargs: Dict[str, Any],
    keys: Union[str, Iterable[str]],
    default: Any = "__default",
    pop: bool = False,
) -> Any:
    if isinstance(keys, str):
        return kwargs.get(keys, default)
    for key in keys:
        if key in kwargs:
            if pop:
                return kwargs.pop(key)
            return kwargs.get(key)
    if default != "__default":
        return default
    raise KeyError(f"None of {keys} is found in {kwargs}.")


def attr_is_value(obj: Any, attr: str, value: Any) -> bool:
    if not hasattr(obj, attr):
        return False
    return getattr(obj, attr) == value


def popattr(obj: Any, attr: str, default: Any = "__default") -> Any:
    if not hasattr(obj, attr):
        if default == "__default":
            raise AttributeError(f"'{obj}' object has no attribute '{attr}'")
        return default
    value = getattr(obj, attr)
    delattr(obj, attr)
    return value


def search_attr(obj: Any, keys: Union[str, Iterable[str]]) -> Any:
    attr = None
    if isinstance(keys, str):
        return getattr(obj, keys, None)
    for key in keys:
        attr = getattr(obj, key, None)
        if attr is not None:
            break
    return attr


def keep_setattr(
    obj: Any,
    attr: str,
    value: Any,
    keep_attr: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Set `value` to `attr` and keep the original value to a new attribute
    `keep_attr`.
    If `overwrite` is True, overwrite value of `keep_attr` if it already exists.

    Args:
        obj: The object to set attribute.
        attr: The attribute to set.
        value: The value to set.
        keep_attr: The attribute to keep the original value.
            If `None`, the original value will be kept in "`attr`_orig".
            So if `attr` is "weight" and use `torch.nn.utils.spectral_norm` or
            `torch.nn.utils.weight_norm` which are using "weight_orig" to keep
            the original weight, should not declare `keep_attr` as "weight_orig".
    """
    if not hasattr(obj, attr):
        raise AttributeError(f"'{obj}' object has no attribute '{attr}'")
    if keep_attr is None:
        keep_attr = f"{attr}_orig"
    if not hasattr(obj, keep_attr) or overwrite:
        setattr(obj, keep_attr, getattr(obj, attr))
    setattr(obj, attr, value)


def normalize_int_tuple(value: Union[INT, Tensor], n: int) -> Tuple[int, ...]:
    if isinstance(value, Tensor):
        numel = value.numel()
        if numel != 1:
            raise ValueError(f"Expected a tensor with 1 element, got {numel}")
        value = cast(int, value.item())

    if isinstance(value, int):
        return (value,) * n

    if isinstance(value, map):
        value = tuple(value)

    if not all(isinstance(v, int) for v in value):
        raise TypeError(f"Expected int elements, got {value}.")

    if len(value) == 1 or len(set(value)) == 1:
        return (value[0],) * n

    if len(value) != n:
        raise ValueError(
            f"The argument must be a tuple of {n} integers, got {value}."
        )
    return tuple(value)


def normalize_float_tuple(
    value: Union[FLOAT, INT, Tensor], n: int
) -> Tuple[float, ...]:
    if isinstance(value, Tensor):
        numel = value.numel()
        if numel != 1:
            raise ValueError(f"Expected a tensor with 1 element, got {numel}")
        value = value.item()

    if isinstance(value, (int, float)):
        return (float(value),) * n

    if isinstance(value, map):
        value = tuple(value)

    if not all(isinstance(v, (int, float)) for v in value):
        raise TypeError(f"Expected float elements, got {value}.")

    if len(value) == 1 or len(set(value)) == 1:
        return (float(value[0]),) * n

    if len(value) != n:
        raise ValueError(
            f"The argument must be a tuple of {n} floats, got {value}."
        )
    return tuple(map(float, value))


def makedirs(*path: str) -> str:
    path = os.path.normpath(os.path.join(*path))
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
    return path


def dir_basename(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def file_extension(file: str) -> str:
    return os.path.splitext(file)[-1].lower()


@overload
def normalize_extension(extension: Optional[str]) -> str:
    ...


@overload
def normalize_extension(extension: Sequence[str]) -> Sequence[str]:
    ...


def normalize_extension(
    extension: Optional[STR],
) -> Union[str, Sequence[str]]:
    if extension is None:
        return ".*"
    if isinstance(extension, str):
        extension = extension.lstrip("*").lower()
        if not extension.startswith("."):
            extension = "." + extension
        return extension
    return tuple(map(normalize_extension, extension))


def get_files(
    path: str,
    extensions: Optional[STR] = None,
    sort_key: Optional[Callable] = None,
) -> List[str]:
    extensions = cast(
        Union[str, Tuple[str, ...]], normalize_extension(extensions)
    )
    files = []
    for file in glob.glob(os.path.join(path, "**", "*.*"), recursive=True):
        if file.lower().endswith(extensions):
            files.append(file)
    files.sort(key=sort_key)
    return files


def get_last_file(
    directory: str,
    regex: str = "*",
    key: Optional[Union[Callable, str]] = None,
) -> str:
    directory = os.path.expanduser(directory)
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} not found.")

    files = [
        f
        for f in glob.glob(os.path.join(directory, regex), recursive=False)
        if os.path.isfile(f)
    ]
    if not files:
        raise FileNotFoundError(f"No files found in {directory}.")

    if key is None:
        return max(files)
    if key == "time":
        return max(files, key=os.path.getctime)
    if isinstance(key, str):
        raise ValueError(f"Invalid key: {key}")
    return max(files, key=key)


def highest_power_of_2(n: int) -> int:
    return 2 ** (n.bit_length() - 1)


def maximum_multiple_of_divisor(n: int, divisor: int) -> int:
    """
    Return the maximum multiple of `divisor` that is less than or equal to `n`.
    If `divisor` is larger than `n`, return 0.
    """
    return n - (n % divisor)


def squared_number(n: Union[int, float]) -> int:
    """
    Return square root of `n` if `n` is a perfect square using Babylonian
    algorithm. `n` must be a positive integer. Otherwise, if `n` is a positive
    float number, it must have zero decimal places.

    Returns:
        The square root of `n` if is a perfect square, otherwise 0.
    """
    if n < 1:
        raise ValueError("`n` must be a positive integer.")
    if isinstance(n, float):
        if not cast(float, n).is_integer():
            return 0
        n = int(n)

    x = int(n // 2)
    if x == 0:
        if n == 1:
            return 1
        return 0

    cache = {x}
    while x * x != n:
        x = (x + (n // x)) // 2
        if x in cache:
            return 0
        cache.add(x)
    return x
