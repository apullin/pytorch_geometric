import torch

import torch_geometric.typing
from torch_geometric.nn import ClusterGCNConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


def test_cluster_gcn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = ClusterGCNConv(16, 32, diag_lambda=1.)
    assert str(conv) == 'ClusterGCNConv(16, 32, diag_lambda=1.0)'
    out = conv(x, edge_index)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out, atol=1e-5)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj2.t()), out, atol=1e-5)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj2.t()), out, atol=1e-5)


def test_cluster_gcn_conv_no_self_loops_and_edge_order():
    conv = ClusterGCNConv(4, 4, diag_lambda=1.0, add_self_loops=False).eval()
    x = torch.randn(2, 4)
    assert conv(x, torch.tensor([[0], [1]])).size() == (2, 4)

    ei_a = torch.tensor([[0, 1, 1], [0, 0, 1]])
    ei_b = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert torch.allclose(conv(x, ei_a), conv(x, ei_b), atol=1e-6)

