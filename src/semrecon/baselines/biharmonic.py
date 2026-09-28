"""Biharmonic inpainting by a direct sparse solve.

Let L be the 5-point graph Laplacian on the H x W pixel grid with Neumann
(reflecting) boundaries, i.e. L = D^T D with D the forward-difference
operator. The biharmonic interpolant minimises the discrete thin-plate
energy ||L x||^2 subject to x_K = y_K on the measured set K. Its normality
conditions on the unknown set U are

    (L^2)_UU x_U = -(L^2)_UK y_K ,

a sparse SPD system (13-point stencil) solved here with a direct LU
factorisation. Measured pixels are interpolated exactly, so the method does
no denoising: shot noise at K is passed through unchanged.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _diff_1d(n: int) -> sp.csr_matrix:
    """(n-1) x n forward difference."""
    return sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format="csr")


def neumann_laplacian(H: int, W: int) -> sp.csr_matrix:
    """Positive semidefinite graph Laplacian D^T D on an H x W grid (row-major)."""
    Dh, Dw = _diff_1d(H), _diff_1d(W)
    Lh = (Dh.T @ Dh).tocsr()
    Lw = (Dw.T @ Dw).tocsr()
    return (sp.kron(Lh, sp.identity(W)) + sp.kron(sp.identity(H), Lw)).tocsr()


_LAP_CACHE: dict[tuple[int, int], sp.csr_matrix] = {}


def _bilaplacian(H: int, W: int) -> sp.csr_matrix:
    key = (H, W)
    if key not in _LAP_CACHE:
        L = neumann_laplacian(H, W)
        _LAP_CACHE[key] = (L @ L).tocsr()
    return _LAP_CACHE[key]


def biharmonic_inpaint(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H, W = y.shape
    if mask.all():
        return y.astype(np.float64)
    if not mask.any():
        return np.zeros_like(y, dtype=np.float64)
    B = _bilaplacian(H, W)
    k = mask.ravel()
    u = ~k
    A = B[u][:, u].tocsc()
    rhs = -(B[u][:, k] @ y.ravel()[k].astype(np.float64))
    x = y.astype(np.float64).ravel().copy()
    x[u] = spla.spsolve(A, rhs, permc_spec="COLAMD")
    return x.reshape(H, W)
