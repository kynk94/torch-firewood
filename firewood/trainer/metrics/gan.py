import torch
import torch.nn.functional as F
from torch import Tensor


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
