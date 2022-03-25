from torch import Tensor

def bias_act(
    x: Tensor,
    b: Tensor,
    xref: Tensor,
    yref: Tensor,
    dy: Tensor,
    grad: int,
    dim: int,
    act: int,
    alpha: float,
    gain: float,
    clamp: float,
) -> Tensor: ...
def upfirdn2d(
    x: Tensor,
    f: Tensor,
    upx: int,
    upy: int,
    downx: int,
    downy: int,
    padx0: int,
    padx1: int,
    pady0: int,
    pady1: int,
    flip: bool,
    gain: float,
) -> Tensor: ...
