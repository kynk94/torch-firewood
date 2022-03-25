import pytest
import torch
import torch.nn.functional as F

from firewood.layers.denormalizations import AdaptiveNorm
from firewood.layers.linear import Linear
from tests.helpers.utils import gen_params


@pytest.mark.parametrize(*gen_params("unbiased", [True, False]))
def test_adain(unbiased: bool):
    eps = 1e-9
    B = 2
    num_features = 3
    x_embedding_size = 10
    m_embedding_size = 10

    adain = AdaptiveNorm(num_features, unbiased=unbiased, eps=eps)

    x = torch.randn(B, num_features, x_embedding_size, x_embedding_size)
    m = torch.randn(B, num_features, m_embedding_size, m_embedding_size)

    x_modulated_custom = adain(x, m)

    x_var, x_mean = torch.var_mean(
        x, dim=tuple(range(2, x.ndim)), unbiased=unbiased, keepdim=True
    )
    x_std = (x_var + eps).sqrt()
    m_var, m_mean = torch.var_mean(
        m, dim=tuple(range(2, x.ndim)), unbiased=unbiased, keepdim=True
    )
    m_std = (m_var + eps).sqrt()

    x_normalized = (x - x_mean) / x_std
    x_modulated = x_normalized * m_std + m_mean

    assert torch.allclose(
        x_modulated, x_modulated_custom, atol=1e-7
    ), f"Foward result mismatch. l1: {F.l1_loss(x_modulated, x_modulated_custom)}"


@pytest.mark.parametrize(*gen_params("unbiased", [True, False]))
def test_adain_projection(unbiased: bool):
    eps = 1e-9
    B = 2
    num_features = 3
    modulation_features = 5
    embedding_size = 10

    adain = AdaptiveNorm(
        num_features,
        unbiased=unbiased,
        eps=eps,
        use_projection=True,
        modulation_features_shape=(
            modulation_features,
            embedding_size,
            embedding_size,
        ),
    )
    linear = Linear(modulation_features, num_features * 2, bias=True)
    linear.weight.data = adain.linear.weight.data
    linear.bias.data = adain.linear.bias.data

    x = torch.randn(B, num_features, embedding_size, embedding_size)
    m = torch.randn(B, modulation_features, embedding_size, embedding_size)

    x_modulated_custom = adain(x, m)

    x_var, x_mean = torch.var_mean(
        x, dim=tuple(range(2, x.ndim)), unbiased=unbiased, keepdim=True
    )
    x_std = (x_var + eps).sqrt()

    m_std, m_mean = linear(m.view(m.size(0), -1)).chunk(2, dim=1)
    for _ in range(x.ndim - m_std.ndim):
        m_std = m_std.unsqueeze(-1)
        m_mean = m_mean.unsqueeze(-1)
    m_std = m_std + 1.0

    x_normalized = (x - x_mean) / x_std
    x_modulated = x_normalized * m_std + m_mean

    assert torch.allclose(
        x_modulated, x_modulated_custom, atol=1e-6
    ), f"Foward result mismatch. l1: {F.l1_loss(x_modulated, x_modulated_custom)}"
