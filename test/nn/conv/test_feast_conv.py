import torch

import torch_geometric.typing
from torch_geometric.nn import FeaStConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_feast_conv():
    x1 = torch.randn(4, 16)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = FeaStConv(16, 32, heads=2)
    assert str(conv) == 'FeaStConv(16, 32, heads=2)'

    out = conv(x1, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x1, edge_index), out, atol=1e-6)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, adj2.t()), out, atol=1e-6)

    # Test bipartite message passing:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 2))

    out = conv((x1, x2), edge_index)
    assert out.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 2))
        assert torch.allclose(conv((x1, x2), adj2.t()), out, atol=1e-6)

    if is_full_test():
        assert torch.allclose(jit((x1, x2), edge_index), out, atol=1e-6)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit((x1, x2), adj2.t()), out, atol=1e-6)

    conv_no_sl = FeaStConv(16, 32, heads=2, add_self_loops=False)
    empty_ei = torch.empty((2, 0), dtype=torch.long)
    assert conv_no_sl(torch.randn(4, 16), empty_ei).size() == (4, 32)

