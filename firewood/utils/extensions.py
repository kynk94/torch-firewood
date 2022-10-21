import glob
import hashlib
import importlib
import os
import platform
import shutil
import warnings
from collections import defaultdict
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Union

from torch.utils import cpp_extension

from firewood.common.backend import runtime_build
from firewood.common.types import STR

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CUDA_SOURCES = sorted(
    glob.glob(os.path.join(_PROJECT_ROOT, "csrc", "**", "*.c*"), recursive=True)
)
_CACHED_EXTENSIONS: Dict[str, ModuleType] = dict()


def user_cache_dir(appname: str = "torch_extensions"):
    system = platform.system().lower()
    if system == "windows":
        raise ImportError(f"Please manually clear {appname} cache.")
    if system == "darwin":
        path = os.path.expanduser("~/Library/Caches")
    else:
        path = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    if appname:
        path = os.path.join(path, appname)
    return path


def _find_windows_compiler() -> Union[str, None]:
    compiler_paths = [
        "C:/Program Files (x86)/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64",
        "C:/Program Files (x86)/Microsoft Visual Studio */vc/bin",
    ]
    for compiler_path in compiler_paths:
        found_paths = glob.glob(compiler_path)
        if found_paths:
            return sorted(found_paths)[-1]
    return None


def load_extension(
    sources: STR = _CUDA_SOURCES,
    extra_cuda_cflags: List[str] = ["--use_fast_math"],
    **kwargs: Any,
) -> ModuleType:
    if isinstance(sources, str):
        sources = [sources]
    sources = sorted(os.path.abspath(source) for source in set(sources))
    encoded_sources = ",".join(sources).encode("utf-8")
    extension_hash = hashlib.sha256(encoded_sources).hexdigest()
    if sources == _CUDA_SOURCES:
        name = "_C"
    else:
        name = f"extension_{extension_hash}"
    if name in _CACHED_EXTENSIONS:
        return _CACHED_EXTENSIONS[name]

    print("Loading CPP extension...")

    if platform.system() == "Windows" and not is_success(
        os.system("where cl.exe >nul 2>nul")
    ):
        found_compiler = _find_windows_compiler()
        if found_compiler is None:
            raise RuntimeError(
                "Cannot find compiler like MSVC/GCC."
                "Make sure it's installed and in the PATH"
            )
        os.environ["PATH"] = found_compiler + ";" + os.environ["PATH"]

    try:
        module = cpp_extension.load(
            name=name,
            sources=sources,
            is_python_module=True,
            extra_cuda_cflags=extra_cuda_cflags,
            **kwargs,
        )
    except ImportError:
        cache_dir = user_cache_dir()
        shutil.rmtree(cache_dir, ignore_errors=True)
        module = cpp_extension.load(
            name=name,
            sources=sources,
            is_python_module=True,
            extra_cuda_cflags=extra_cuda_cflags,
            **kwargs,
        )
    try:
        module = importlib.import_module(name)
    except Exception:
        raise RuntimeError(
            f"Successfully compiled, but failed to import extension."
        )

    _CACHED_EXTENSIONS[name] = module
    return module


def is_success(status: int) -> bool:
    return status == 0


class CUDAExtension:
    __C = None
    __use_prebuild = True
    __cache: Dict[str, Dict[Any, Any]] = defaultdict(dict)
    cuda_cflags = ["--use_fast_math"]

    @classmethod
    def import_C(cls) -> None:
        if runtime_build():
            cls.__import_runtime_build()
        else:
            cls.__import_prebuild()

    @classmethod
    def available_operations(cls) -> List[str]:
        cls.import_C()
        return [o for o in dir(cls.__C) if not o.startswith("__")]

    @classmethod
    def cache(cls, name: Optional[str] = None) -> Dict[Any, Any]:
        if runtime_build() and cls.__use_prebuild:
            cls.__set_prebuild(False)
        if name is None:
            return cls.__cache
        return cls.__cache[name]

    @classmethod
    def get(cls, name: str) -> Callable[..., Any]:
        cls.import_C()
        operation = getattr(cls.__C, name, None)
        if operation is None:
            support_operations = ", ".join(cls.available_operations())
            raise AttributeError(
                f"{name} is not found in CUDA extension. "
                f"Available operations: {support_operations}"
            )
        return operation

    @classmethod
    def __clear_cache(cls) -> None:
        for value in cls.__cache.values():
            value.clear()

    @classmethod
    def __set_prebuild(cls, use_prebuild: bool) -> None:
        if cls.__use_prebuild != use_prebuild:
            cls.__clear_cache()
        cls.__use_prebuild = use_prebuild

    @classmethod
    def __import_runtime_build(
        cls, cuda_cflags: Optional[List[str]] = None
    ) -> ModuleType:
        _C = load_extension(
            sources=_CUDA_SOURCES,
            extra_cuda_cflags=cuda_cflags or cls.cuda_cflags,
        )
        if cls.__C is not None and cls.__C != _C:
            cls.__clear_cache()
        cls.__C = _C
        cls.__set_prebuild(False)
        return cls.__C

    @classmethod
    def __import_prebuild(cls) -> ModuleType:
        try:
            import firewood._C

            if cls.__C is not None and cls.__C != firewood._C:
                cls.__clear_cache()
            cls.__C = firewood._C
            cls.__set_prebuild(True)
        except (RuntimeError, ModuleNotFoundError):
            warnings.warn(
                "Pre-build CUDA extension not found. "
                "Build CUDA extension now."
            )
            return cls.__import_runtime_build()
        return cls.__C
