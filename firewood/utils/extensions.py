import glob
import importlib
import os
import platform
from types import ModuleType
from typing import Any, Dict, List, Union

from torch.utils import cpp_extension

PLATFORM: str = platform.system()
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_CUDA_SOURCES = glob.glob(
    os.path.join(_PROJECT_ROOT, "csrc", "**", "*.c*"), recursive=True
)
_CACHED_EXTENSIONS: Dict[str, ModuleType] = dict()


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


def load_cuda_extension(
    extension: str = "_C",
    sources: List[str] = _CUDA_SOURCES,
    extra_cuda_cflags: List[str] = ["--use_fast_math"],
    **kwargs: Any,
) -> ModuleType:
    if extension in _CACHED_EXTENSIONS:
        return _CACHED_EXTENSIONS[extension]

    if PLATFORM == "Windows" and not is_success(
        os.system("where cl.exe >nul 2>nul")
    ):
        found_compiler = _find_windows_compiler()
        if found_compiler is None:
            raise RuntimeError(
                "Cannot find compiler like MSVC/GCC."
                "Make sure it's installed and in the PATH"
            )
        os.environ["PATH"] = found_compiler + ";" + os.environ["PATH"]

    module = cpp_extension.load(
        name=extension,
        sources=sources,
        is_python_module=True,
        extra_cuda_cflags=extra_cuda_cflags,
        **kwargs,
    )
    try:
        module = importlib.import_module(extension)
    except Exception:
        print(f"Successfully compiled, but failed to import {extension}")

    _CACHED_EXTENSIONS[extension] = module
    return module


def is_success(status: int) -> bool:
    return status == 0
