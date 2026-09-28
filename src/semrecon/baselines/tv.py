r"""Total-variation inpainting, solved from scratch.

Problem
-------
With isotropic TV, TV(x) = sum_i ||(grad x)_i||_2, and a data term G over the
measured set M:

  TV-L2   min_x  lam * TV(x) + 1/2 sum_{i in M} (x_i - y_i)^2
  TV-KL   min_x  lam * TV(x) + sum_{i in M} (x_i - y_i log x_i)

TV-KL is the Poisson negative log-likelihood (divided by the dose D): with
counts n_i = D y_i ~ Poisson(D x_i), -log p = D x_i - n_i log x_i + const.
Both optionally include the box constraint 0 <= x <= 1.

Discretisation
--------------
grad uses forward differences with Neumann boundaries (the last difference in
each direction is zero); div = -grad^T, so <grad x, p> = <x, -div p> exactly.
||grad||^2 <= 8.

Solvers
-------
pdhg   Chambolle-Pock primal-dual hybrid gradient on
         min_x F(K x) + G(x),  K = grad,  F = lam ||.||_{2,1}
       Iteration (sigma * tau * 8 < 1):
         p     <- proj_{||p_i|| <= lam}(p + sigma grad xbar)      (prox of F*)
         x_new <- prox_{tau G}(x + tau div p)
         xbar  <- 2 x_new - x
       prox_{tau G} is pixelwise and closed-form:
         L2:  (v + tau y) / (1 + tau)                      on M,  v off M
         KL:  ((v - tau) + sqrt((v - tau)^2 + 4 tau y)) / 2 on M,  v off M
       followed by clipping to [0, 1] when box=True (exact for a 1-D convex
       function plus an interval indicator).

admm   ADMM on the split z = grad x (TV-L2 only, no box):
         x <- argmin 1/2||M(x - y)||^2 + rho/2 ||grad x - z + u||^2
              i.e.  (M + rho grad^T grad) x = M y + rho grad^T (z - u)
         z <- group soft-threshold(grad x + u, lam / rho)
         u <- u + grad x - z
       The mask makes the x-system non-diagonalisable by any fast transform,
       so it is solved inexactly by warm-started preconditioned CG. The
       preconditioner replaces M by its mean f (the sampling fraction), giving
       (f I + rho grad^T grad)^{-1}, which *is* diagonal in the DCT-II basis
       (grad^T grad is the Neumann Laplacian, eigenvalues
       (2 - 2 cos(pi k / H)) + (2 - 2 cos(pi l / W))).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import dctn, idctn


# ---------------------------------------------------------------- operators

def grad(x: np.ndarray) -> np.ndarray:
    """Forward differences, Neumann boundary. (H, W) -> (2, H, W)."""
    g = np.zeros((2,) + x.shape, dtype=x.dtype)
    g[0, :-1] = x[1:] - x[:-1]
    g[1, :, :-1] = x[:, 1:] - x[:, :-1]
    return g


def div(p: np.ndarray) -> np.ndarray:
    """Negative adjoint of grad. (2, H, W) -> (H, W)."""
    d = np.zeros(p.shape[1:], dtype=p.dtype)
    d[:-1] += p[0, :-1]
    d[1:] -= p[0, :-1]
    d[:, :-1] += p[1, :, :-1]
    d[:, 1:] -= p[1, :, :-1]
    return d


def tv(x: np.ndarray) -> float:
    return float(np.sqrt((grad(x) ** 2).sum(0)).sum())


def _proj_ball(p: np.ndarray, radius: float) -> np.ndarray:
    n = np.sqrt((p**2).sum(0, keepdims=True))
    return p / np.maximum(1.0, n / radius)


def _shrink(v: np.ndarray, t: float) -> np.ndarray:
    n = np.sqrt((v**2).sum(0, keepdims=True))
    return v * np.maximum(0.0, 1.0 - t / np.maximum(n, 1e-12))


# ---------------------------------------------------------------- objectives

def objective(x: np.ndarray, y: np.ndarray, mask: np.ndarray, lam: float, data: str = "l2") -> float:
    if data == "l2":
        fid = 0.5 * float(((x - y)[mask] ** 2).sum())
    elif data == "kl":
        xm, ym = np.maximum(x[mask], 1e-12), y[mask]
        fid = float((xm - ym * np.log(xm)).sum())
    else:
        raise ValueError(data)
    return lam * tv(x) + fid


@dataclass
class History:
    objective: list[float] = field(default_factory=list)
    rel_change: list[float] = field(default_factory=list)


# ---------------------------------------------------------------- PDHG

def pdhg(
    y: np.ndarray,
    mask: np.ndarray,
    lam: float,
    data: str = "l2",
    box: bool = True,
    max_iter: int = 500,
    tol: float = 1e-5,
    x0: np.ndarray | None = None,
    track: bool = False,
) -> tuple[np.ndarray, History]:
    y = y.astype(np.float64)
    m = mask.astype(bool)
    tau = sigma = 0.99 / np.sqrt(8.0)
    x = (y.copy() if x0 is None else x0.astype(np.float64).copy())
    xbar = x.copy()
    p = np.zeros((2,) + y.shape)
    hist = History()
    for _ in range(max_iter):
        p = _proj_ball(p + sigma * grad(xbar), lam)
        v = x + tau * div(p)
        x_new = v.copy()
        if data == "l2":
            x_new[m] = (v[m] + tau * y[m]) / (1.0 + tau)
        elif data == "kl":
            a = v[m] - tau
            x_new[m] = 0.5 * (a + np.sqrt(a * a + 4.0 * tau * y[m]))
        else:
            raise ValueError(data)
        if box:
            np.clip(x_new, 0.0, 1.0, out=x_new)
        rel = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-12)
        xbar = 2.0 * x_new - x
        x = x_new
        if track:
            hist.objective.append(objective(x, y, m, lam, data))
            hist.rel_change.append(float(rel))
        if rel < tol:
            break
    return x, hist


def nearest_fill(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy each unmeasured pixel from its nearest measured pixel (EDT)."""
    from scipy.ndimage import distance_transform_edt

    idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return y[tuple(idx)]


def tv_inpaint(y: np.ndarray, mask: np.ndarray, lam: float, data: str = "l2", **kw) -> np.ndarray:
    """Baseline entry point: PDHG warm-started from a nearest-neighbour fill.

    From a zero start, TV propagates information into a gap by roughly one
    pixel per iteration, so large gaps (partial raster at 5%) would need
    hundreds of iterations just to be reached.
    """
    kw.setdefault("x0", nearest_fill(y, mask))
    return pdhg(y, mask, lam, data=data, **kw)[0]


# ---------------------------------------------------------------- ADMM

def _laplacian_eigs(H: int, W: int) -> np.ndarray:
    kh = 2.0 - 2.0 * np.cos(np.pi * np.arange(H) / H)
    kw = 2.0 - 2.0 * np.cos(np.pi * np.arange(W) / W)
    return kh[:, None] + kw[None, :]


def _pcg(apply_A, b, x, apply_Minv, iters: int, tol: float = 1e-8):
    r = b - apply_A(x)
    z = apply_Minv(r)
    d = z.copy()
    rz = (r * z).sum()
    bn = np.linalg.norm(b)
    for _ in range(iters):
        if np.linalg.norm(r) <= tol * bn:
            break
        Ad = apply_A(d)
        alpha = rz / (d * Ad).sum()
        x = x + alpha * d
        r = r - alpha * Ad
        if np.linalg.norm(r) <= tol * bn:
            break
        z = apply_Minv(r)
        rz_new = (r * z).sum()
        d = z + (rz_new / rz) * d
        rz = rz_new
    return x


def admm(
    y: np.ndarray,
    mask: np.ndarray,
    lam: float,
    rho: float = 1.0,
    max_iter: int = 500,
    cg_iters: int = 5,
    tol: float = 1e-5,
    precondition: bool = True,
    track: bool = False,
) -> tuple[np.ndarray, History]:
    y = y.astype(np.float64)
    m = mask.astype(np.float64)
    H, W = y.shape
    f = max(m.mean(), 1e-3)
    eig = f + rho * _laplacian_eigs(H, W)

    def apply_A(v):
        return m * v - rho * div(grad(v))

    if precondition:
        def apply_Minv(r):
            return idctn(dctn(r, type=2, norm="ortho") / eig, type=2, norm="ortho")
    else:
        def apply_Minv(r):
            return r

    x = y.copy()
    z = grad(x)
    u = np.zeros_like(z)
    my = m * y
    hist = History()
    for _ in range(max_iter):
        b = my - rho * div(z - u)
        x_new = _pcg(apply_A, b, x, apply_Minv, cg_iters)
        gx = grad(x_new)
        z = _shrink(gx + u, lam / rho)
        u = u + gx - z
        rel = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-12)
        primal_res = np.linalg.norm(gx - z) / max(np.linalg.norm(gx), 1e-12)
        x = x_new
        if track:
            hist.objective.append(objective(x, y, mask.astype(bool), lam, "l2"))
            hist.rel_change.append(float(rel))
        if rel < tol and primal_res < tol:
            break
    return x, hist
