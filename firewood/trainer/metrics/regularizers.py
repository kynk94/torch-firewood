import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric

from firewood.common.backend import no_weight_grad_in_gfix_conv
from firewood.utils import get_nested_first


class PathLengthPenalty(Metric):
    """
    Path length regularizer for stylized GANs.
    Introduced in StyleGAN2.
    """

    is_differentiable = True
    full_state_update = False
    path_length: Tensor
    path_length_avg: Tensor

    def __init__(self, decay: float = 0.99, eps: float = 1e-8) -> None:
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.add_state(
            "path_length", default=torch.tensor(0.0), dist_reduce_fx="mean"
        )
        self.add_state(
            "path_length_avg",
            default=torch.tensor(0.0),
            dist_reduce_fx="mean",
            persistent=True,
        )

    @no_weight_grad_in_gfix_conv()
    def update(self, styles: Tensor, generated_images: Tensor) -> None:
        """
        Args:
            styles: (batch_size, num_layers, num_channels)
            generated_images: (batch_size, num_channels, height, width)
        """
        H, W = generated_images.shape[-2:]
        images = generated_images
        noise: Tensor = torch.randn_like(images) / np.sqrt(H * W)
        grad = torch.autograd.grad(
            outputs=(images * noise).sum(), inputs=styles, create_graph=True
        )[0]
        if grad.ndim == 2:
            path_length = grad.square().mean(-1).sqrt()
        else:
            path_length = grad.square().sum(-1).mean(-1).sqrt()
        self.path_length = path_length
        path_length_avg = self.path_length_avg.lerp(
            path_length.mean(), 1.0 - self.decay
        )
        self.path_length_avg.copy_(path_length_avg.detach())

    def compute(self) -> Tensor:
        return (self.path_length - self.path_length_avg).square().mean()

    def reset(self) -> None:
        path_length_avg = self.path_length_avg.detach()
        super().reset()
        self.path_length_avg.copy_(path_length_avg)


def gradient_penalty(
    discriminator: nn.Module,
    real_images: Tensor,
    fake_images: Tensor,
) -> Tensor:
    alpha_size = (real_images.size(0),) + (1,) * (real_images.ndim - 1)
    alpha = torch.rand(alpha_size, device=real_images.device)
    interpolates: Tensor = alpha * real_images + (1 - alpha) * fake_images
    interpolates.requires_grad = True
    score: Tensor = get_nested_first(discriminator(interpolates))
    gradients = torch.autograd.grad(
        outputs=score.mean(),
        inputs=interpolates,
        create_graph=True,
    )[0]
    # norm: (B,). There are no non-batch axes that need to be summed.
    norm: Tensor = gradients.norm(2, dim=tuple(range(1, gradients.ndim)))
    return norm.sub(1.0).square().mean()


def simple_gradient_penalty(
    score: Tensor,
    images: Tensor,
) -> Tensor:
    gradients = torch.autograd.grad(
        outputs=score.mean(),
        inputs=images,
        create_graph=True,
    )[0]
    return gradients.square().sum(tuple(range(1, gradients.ndim))).mean()
