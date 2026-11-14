from typing import Optional

import torch
from torch import Tensor


class SGFormerAttention(torch.nn.Module):
    r"""The simple global attention mechanism from the
    `"SGFormer: Simplifying and Empowering Transformers for
    Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of parallel attention heads.
            (default: :obj:`1.`)
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64.`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
    """
    def __init__(
        self,
        channels: int,
        heads: int = 1,
        head_channels: int = 64,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert channels % heads == 0
        if head_channels is None:
            head_channels = channels // heads

        self.heads = heads
        self.head_channels = head_channels

        inner_channels = head_channels * heads
        self.q = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = torch.nn.Linear(channels, inner_channels, bias=qkv_bias)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        B, N, *_ = x.shape
        qs, ks, vs = self.q(x), self.k(x), self.v(x)
        # reshape and permute q, k and v to proper shape
        # (b, n, num_heads * head_channels) to (b, n, num_heads, head_channels)
        qs, ks, vs = map(
            lambda t: t.reshape(B, N, self.heads, self.head_channels),
            (qs, ks, vs))

        # replace 0's with epsilon
        epsilon = 1e-6
        qs[qs == 0] = epsilon
        ks[ks == 0] = epsilon
        # normalize input, shape not changed
        qs, ks = map(
            lambda t: t / torch.linalg.norm(t, ord=2, dim=-1, keepdim=True),
            (qs, ks))

        if mask is not None:
            mask_4d = mask[:, :, None, None]
            ks = ks.masked_fill(~mask_4d, 0.)
            vs = vs.masked_fill(~mask_4d, 0.)
            num_valid = mask.sum(dim=1).view(B, 1, 1, 1).to(vs.dtype).clamp(min=1.0)
        else:
            num_valid = float(N)

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)
        attention_num += num_valid * vs

        # denominator
        all_ones = torch.ones([B, N], dtype=ks.dtype, device=ks.device)
        ks_sum = torch.einsum("blhm,bl->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)
        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer,
                                               len(attention_normalizer.shape))
        attention_normalizer += num_valid
        attn_output = attention_num / attention_normalizer

        out = attn_output.mean(dim=2)
        if mask is not None:
            out = out.masked_fill(~mask[:, :, None], 0.)
        return out

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'heads={self.heads}, '
                f'head_channels={self.head_channels})')
