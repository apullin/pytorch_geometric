import warnings

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import EdgeIndex
from torch_geometric.typing import Adj, SparseTensor, torch_sparse
from torch_geometric.utils import is_torch_sparse_tensor, scatter


def spmm(
    src: Adj,
    other: Tensor,
    reduce: str = 'sum',
) -> Tensor:
    r"""Matrix product of sparse matrix with dense matrix.

    Args:
        src (torch.Tensor or torch_sparse.SparseTensor or EdgeIndex):
            The input sparse matrix which can be a
            :pyg:`PyG` :class:`torch_sparse.SparseTensor`,
            a :pytorch:`PyTorch` :class:`torch.sparse.Tensor` or
            a :pyg:`PyG` :class:`EdgeIndex`.
        other (torch.Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    reduce = 'sum' if reduce == 'add' else reduce

    if reduce not in ['sum', 'mean', 'min', 'max']:
        raise ValueError(f"`reduce` argument '{reduce}' not supported")

    if other.dim() > 2:
        orig_shape = list(other.size())
        num_src_nodes = orig_shape[-2]
        feat_dim = orig_shape[-1]
        if reduce in ['sum', 'mean']:
            perm_other = torch.movedim(other, -2, 0).reshape(num_src_nodes, -1)
            out_2d = _spmm_2d(src, perm_other, reduce=reduce)
            out_shape = [out_2d.size(0)] + orig_shape[:-2] + [feat_dim]
            return torch.movedim(out_2d.view(out_shape), 0, -2).contiguous()
        else:
            flat_other = other.reshape(-1, num_src_nodes, feat_dim)
            outs = [_spmm_2d(src, flat_other[b], reduce=reduce)
                    for b in range(flat_other.size(0))]
            stacked = torch.stack(outs, dim=0)
            return stacked.view(orig_shape[:-2] + [stacked.size(-2), feat_dim])

    return _spmm_2d(src, other, reduce=reduce)

def _spmm_2d(
    src: Adj,
    other: Tensor,
    reduce: str = 'sum',
) -> Tensor:
    if not torch.jit.is_scripting() and isinstance(src, EdgeIndex):
        return src.matmul(other=other, reduce=reduce)  # type: ignore

    if isinstance(src, SparseTensor):
        if src.nnz() == 0:
            return other.new_zeros(src.size(0), other.size(1))

        if (torch_geometric.typing.WITH_PT20 and other.dim() == 2
                and not src.is_cuda() and not src.requires_grad()):
            # Use optimized PyTorch `torch.sparse.mm` path:
            csr = src.to_torch_sparse_csr_tensor().to(other.dtype)
            return torch.sparse.mm(csr, other, reduce)
        return torch_sparse.matmul(src, other, reduce)

    if not is_torch_sparse_tensor(src):
        raise ValueError("'src' must be a 'torch_sparse.SparseTensor' or a "
                         "'torch.sparse.Tensor'")

    # `torch.sparse.mm` only supports reductions on CPU for PyTorch>=2.0.
    # This will currently throw on error for CUDA tensors.
    if torch_geometric.typing.WITH_PT20:

        if src.is_cuda and (reduce == 'min' or reduce == 'max'):
            raise NotImplementedError(f"`{reduce}` reduction is not yet "
                                      f"supported for 'torch.sparse.Tensor' "
                                      f"on device '{src.device}'")

        # Always convert COO to CSR for more efficient processing:
        if src.layout == torch.sparse_coo:
            warnings.warn(
                f"Converting sparse tensor to CSR format for more "
                f"efficient processing. Consider converting your "
                f"sparse tensor to CSR format beforehand to avoid "
                f"repeated conversion (got '{src.layout}')", stacklevel=2)
            src = src.to_sparse_csr()

        # Warn in case of CSC format without gradient computation:
        if src.layout == torch.sparse_csc and not other.requires_grad:
            warnings.warn(
                f"Converting sparse tensor to CSR format for more "
                f"efficient processing. Consider converting your "
                f"sparse tensor to CSR format beforehand to avoid "
                f"repeated conversion (got '{src.layout}')", stacklevel=2)

        # Use the default code path for `sum` reduction (works on CPU/GPU):
        if reduce == 'sum':
            return torch.sparse.mm(src, other)

        # Use the default code path with custom reduction (works on CPU):
        if src.layout == torch.sparse_csr and not src.is_cuda:
            return torch.sparse.mm(src, other, reduce)

        # Simulate `mean` reduction by dividing by degree:
        if reduce == 'mean':
            if src.layout == torch.sparse_csr:
                ptr = src.crow_indices()
                deg = ptr[1:] - ptr[:-1]
            else:
                assert src.layout == torch.sparse_csc
                deg = scatter(torch.ones_like(src.values()), src.row_indices(),
                              dim=0, dim_size=src.size(0), reduce='sum')

            return torch.sparse.mm(src, other) / deg.view(-1, 1).clamp_(min=1)

        # TODO The `torch.sparse.mm` code path with the `reduce` argument does
        # not yet support CSC :(
        if src.layout == torch.sparse_csc:
            warnings.warn(
                f"Converting sparse tensor to CSR format for more "
                f"efficient processing. Consider converting your "
                f"sparse tensor to CSR format beforehand to avoid "
                f"repeated conversion (got '{src.layout}')", stacklevel=2)
            src = src.to_sparse_csr()

        return torch.sparse.mm(src, other, reduce)

    # pragma: no cover
    # PyTorch < 2.0 only supports sparse COO format:
    if reduce == 'sum':
        return torch.sparse.mm(src, other)
    elif reduce == 'mean':
        if src.layout == torch.sparse_csr:
            ptr = src.crow_indices()
            deg = ptr[1:] - ptr[:-1]
        elif src.layout == torch.sparse_csc:
            assert src.layout == torch.sparse_csc
            ones = torch.ones_like(src.values())
            index = src.row_indices()
            deg = scatter(ones, index, 0, dim_size=src.size(0), reduce='sum')
        else:
            assert src.layout == torch.sparse_coo
            src = src.coalesce()
            ones = torch.ones_like(src.values())
            index = src.indices()[0]
            deg = scatter(ones, index, 0, dim_size=src.size(0), reduce='sum')

        return torch.sparse.mm(src, other) / deg.view(-1, 1).clamp_(min=1)

    raise ValueError(f"`{reduce}` reduction is not supported for "
                     f"'torch.sparse.Tensor' on device '{src.device}'")
