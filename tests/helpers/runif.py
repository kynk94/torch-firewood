import gc
import sys
from typing import Optional

import pytest
import torch
from packaging.version import Version
from pkg_resources import get_distribution


def runif(
    *args,
    min_gpus: int = 0,
    min_python: Optional[str] = None,
    min_torch: Optional[str] = None,
    max_torch: Optional[str] = None,
    tensorflow_installed: Optional[bool] = None,
):
    conditions = []
    reasons = []

    if min_gpus:
        conditions.append(torch.cuda.device_count() >= min_gpus)
        reasons.append(f"GPUs >= {min_gpus}")
        gc.collect()
        torch.cuda.empty_cache()

    if min_python:
        version = Version(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        min_version = Version(min_python)
        conditions.append(version >= min_version)
        reasons.append(f"Python >= {min_python}")

    if min_torch:
        version = Version(get_distribution("torch").version)
        min_version = Version(min_torch)
        conditions.append(version >= min_version)
        reasons.append(f"torch >= {min_torch}")

    if max_torch:
        version = Version(get_distribution("torch").version)
        max_version = Version(max_torch)
        conditions.append(version <= max_version)
        reasons.append(f"torch <= {max_torch}")

    if tensorflow_installed:
        try:
            import tensorflow as tf

            conditions.append(True)
        except ImportError:
            conditions.append(False)
        reasons.append("tensorflow not installed")

    condition = not all(conditions)
    reason = f"Requires: [{', '.join(reasons)}]"
    return pytest.mark.skipif(*args, condition=condition, reason=reason)
