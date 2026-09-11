"""Minimal shim for the two torch_scatter calls in continuous-remeshing (no prebuilt wheel for
torch 2.12+cu130). Semantics for dim=0 with an explicit `out` tensor, as used in core/opt.py and
core/remesh.py: scatter_mean averages src rows into out[index]; scatter_max writes the running
max into out (out's existing values participate, matching torch_scatter's behaviour when out given)."""
import torch
__version__ = "shim-0.1"

def _expand_index(index, src):
    if index.shape != src.shape:
        index = index.reshape(index.shape[0], *([1] * (src.dim() - 1))).expand_as(src)
    return index

def scatter_mean(src, index, dim=0, out=None, dim_size=None):
    assert dim == 0
    index = _expand_index(index, src)
    if out is None:
        n = dim_size if dim_size is not None else int(index.max()) + 1
        out = torch.zeros((n, *src.shape[1:]), dtype=src.dtype, device=src.device)
    out.scatter_reduce_(0, index, src, reduce="mean", include_self=False)
    return out

def scatter_max(src, index, dim=0, out=None, dim_size=None):
    assert dim == 0
    index = _expand_index(index, src)
    if out is None:
        n = dim_size if dim_size is not None else int(index.max()) + 1
        out = torch.zeros((n, *src.shape[1:]), dtype=src.dtype, device=src.device)
    out.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    return out, None
