from firewood import functional, hooks, layers, models, utils
from firewood.__version__ import __version__
from firewood.common import backend, collector

if utils.is_newer_torch("1.11.0") and utils.is_older_torch("1.13.0"):
    # cudnn_convolution_backward_weight is deprecated since 1.11.0
    # and `torch.grad.convNd_weight` OOM error is fixed since 1.13.0
    import warnings

    warnings.warn(
        "`torch.grad.convNd_weight` has OOM issue on torch<1.13.0, "
        "please upgrade torch to 1.13.0 or higher.\n"
        "Or, calling `firewood.utils.apply.set_all_conv_force_default(True)` "
        "prevents this issue, but may cause operation slower.",
        category=Warning,
    )
