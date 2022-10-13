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


def logistic_saturating_loss(score: Tensor, is_real: bool) -> Tensor:
    """
    return: log(1 - logistic(score))
    """
    if is_real:
        return -F.softplus(-score).mean()
    return -F.softplus(score).mean()


def logistic_nonsaturating_loss(score: Tensor, is_real: bool) -> Tensor:
    """
    return: -log(logistic(score))
    """
    if is_real:
        return F.softplus(-score).mean()
    return F.softplus(score).mean()


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
