from typing import Optional, Tuple, cast

import torch
from torch import Tensor


def print_cuda_memory(input: Optional[Tensor] = None) -> None:
    vram_free, vram_total = cast(
        Tuple[float, float],
        torch.cuda.mem_get_info(torch.cuda.current_device()),
    )
    vram_used = (vram_total - vram_free) / 1024**2
    vram_total = vram_total / 1024**2

    message = "\rCUDA Memory Test - "
    message += f"Memory: {int(vram_used):d} MiB / {int(vram_total):d} MiB\t"
    if input is not None:
        message += f"Input: {input.shape}\t"
    print(message)
