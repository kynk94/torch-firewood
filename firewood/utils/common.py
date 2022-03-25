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

from packaging.version import Version
from pkg_resources import get_distribution

from firewood.common.types import INT, STR


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


def normalize_int_tuple(value: INT, n: int) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value,) * n

    if not all(isinstance(v, int) for v in value):
        raise TypeError(f"Expected int elements, got {value}.")

    if len(value) == 1 or len(set(value)) == 1:
        return (value[0],) * n

    if len(value) != n:
        raise ValueError(
            f"The argument must be a tuple of {n} integers, got {value}."
        )
    return tuple(value)


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


def squared_number(n: Union[int, float]) -> int:
    """
    Return square root of n if n is a perfect square using Babylonian algorithm.
    n must be a positive integer. Otherwise, if n is a positive float number,
    it must have zero decimal places.
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")
    if isinstance(n, float) and not n.is_integer():  # type: ignore
        return 0

    x = n // 2
    if x == 0:
        if n == 1:
            return 1
        return 0

    cache = set((x,))
    while x * x != n:
        x = (x + (n // x)) // 2
        if x in cache:
            return 0
        cache.add(x)
    return int(x)
