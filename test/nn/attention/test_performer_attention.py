import torch

from torch_geometric.nn.attention import PerformerAttention


def test_performer_attention():
    x = torch.randn(1, 4, 16)
    mask = torch.ones([1, 4], dtype=torch.bool)
    attn = PerformerAttention(channels=16, heads=4)
    out = attn(x, mask)
    assert out.shape == (1, 4, 16)
    assert str(attn) == ('PerformerAttention(heads=4, '
                         'head_channels=64 kernel=ReLU())')


def test_linear_attention_mask_padding_invariance():
    from torch_geometric.nn.attention import (
        PolynormerAttention,
        SGFormerAttention,
    )

    x_clean = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, True, True, True],
                         [True, True, False, False, False]])
    x_dirty = x_clean.clone()
    x_dirty[~mask] = 100.0

    for attn in [
        PerformerAttention(channels=16, heads=4, head_channels=4).eval(),
        SGFormerAttention(channels=16, heads=4, head_channels=4).eval(),
        PolynormerAttention(channels=16, heads=4, head_channels=4).eval(),
    ]:
        y_clean = attn(x_clean, mask=mask)
        y_dirty = attn(x_dirty, mask=mask)
        assert torch.allclose(y_clean[mask], y_dirty[mask], atol=1e-5)
        assert float(y_dirty[~mask].abs().max()) == 0.0

