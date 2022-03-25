import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from firewood.utils import get_nested_first


def gan_loss(score: Tensor, is_real: bool, reduction: str = "mean") -> Tensor:
    return F.binary_cross_entropy_with_logits(
        score, torch.full_like(score, is_real), reduction=reduction
    )


def logits_gan_loss(
    logits: Tensor, is_real: bool, reduction: str = "mean"
) -> Tensor:
    return F.cross_entropy(
        logits, torch.full_like(logits, is_real), reduction=reduction
    )


def lsgan_loss(score: Tensor, is_real: bool, reduction: str = "mean") -> Tensor:
    return F.mse_loss(
        score, torch.full_like(score, is_real), reduction=reduction
    )


def wgan_loss(score: Tensor, is_real: bool) -> Tensor:
    if is_real:
        return -score.mean()
    return score.mean()


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
        outputs=score,
        inputs=interpolates,
        grad_outputs=torch.ones_like(score, pin_memory=True),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
