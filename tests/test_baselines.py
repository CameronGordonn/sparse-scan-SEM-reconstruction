import numpy as np
import pytest
from skimage.restoration import denoise_tv_chambolle, inpaint_biharmonic

from semrecon.baselines import tv
from semrecon.baselines.biharmonic import biharmonic_inpaint
from semrecon.metrics import psnr
from semrecon.patterns import make_mask


def smooth_image(n=48, seed=0):
    rng = np.random.default_rng(seed)
    i, j = np.mgrid[0:n, 0:n] / n
    x = 0.5 + 0.2 * np.sin(2 * np.pi * i) * np.cos(3 * np.pi * j) + 0.1 * (i > 0.5)
    return (x + 0.02 * rng.standard_normal((n, n))).clip(0, 1)


def test_grad_div_adjoint():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((17, 23))
    p = rng.standard_normal((2, 17, 23))
    assert np.isclose((tv.grad(x) * p).sum(), -(x * tv.div(p)).sum())


def test_grad_norm_bound():
    # power iteration on div(grad(.)): ||grad||^2 <= 8
    x = np.random.default_rng(0).standard_normal((32, 32))
    for _ in range(200):
        x = -tv.div(tv.grad(x))
        x /= np.linalg.norm(x)
    assert np.linalg.norm(tv.grad(x)) ** 2 <= 8.0


def test_pdhg_matches_skimage_denoising():
    y = smooth_image()
    lam = 0.05
    ours, _ = tv.pdhg(y, np.ones_like(y, bool), lam, box=False, max_iter=5000, tol=1e-10)
    ref = denoise_tv_chambolle(y, weight=lam, eps=1e-10, max_num_iter=20000)
    assert np.abs(ours - ref).max() < 2e-3
    o = tv.objective(ours, y, np.ones_like(y, bool), lam)
    r = tv.objective(ref, y, np.ones_like(y, bool), lam)
    assert o <= r * (1 + 1e-4)


def test_admm_and_pdhg_agree_on_inpainting():
    x = smooth_image()
    m = make_mask("uniform", x.shape, 0.3, np.random.default_rng(0))
    y = x * m
    lam = 0.02
    a, _ = tv.admm(y, m, lam, rho=0.5, max_iter=3000, cg_iters=10, tol=1e-10)
    p, _ = tv.pdhg(y, m, lam, box=False, max_iter=20000, tol=1e-10)
    oa, op = tv.objective(a, y, m, lam), tv.objective(p, y, m, lam)
    assert abs(oa - op) / op < 1e-3


@pytest.mark.parametrize("data", ["l2", "kl"])
def test_pdhg_objective_decreases_overall(data):
    x = smooth_image()
    m = make_mask("line_hop", x.shape, 0.2, np.random.default_rng(0), mean_segment=6)
    y = np.random.default_rng(1).poisson(x * 30) / 30.0 * m
    _, h = tv.pdhg(y, m, 0.05, data=data, max_iter=400, tol=0, track=True)
    obj = np.array(h.objective)
    assert obj[-1] < obj[0]
    # tail is (nearly) monotone: PDHG is not a descent method, but must settle
    assert (np.diff(obj[200:]) <= 1e-6 * abs(obj[-1])).mean() > 0.95


def test_kl_prox_optimality():
    # x = prox_{tau (x - y log x)}(v) solves 1 - y/x + (x - v)/tau = 0
    rng = np.random.default_rng(0)
    v, y, tau = rng.standard_normal(100), rng.random(100), 0.3
    a = v - tau
    x = 0.5 * (a + np.sqrt(a * a + 4 * tau * y))
    assert np.allclose(1 - y / x + (x - v) / tau, 0, atol=1e-10)


def test_tv_inpaint_improves_on_zero_fill():
    x = smooth_image(64)
    m = make_mask("partial_raster", x.shape, 0.2, np.random.default_rng(0))
    r = tv.tv_inpaint(x * m, m, 0.01)
    assert psnr(r, x) > psnr(x * m, x) + 10


def test_biharmonic_interpolates_and_matches_skimage():
    x = smooth_image(40)
    m = make_mask("uniform", x.shape, 0.2, np.random.default_rng(0))
    ours = biharmonic_inpaint(x * m, m)
    assert np.allclose(ours[m], x[m])
    ref = inpaint_biharmonic(x, ~m)
    assert psnr(np.clip(ours, 0, 1), np.clip(ref, 0, 1)) > 40


def test_biharmonic_reproduces_constants():
    m = make_mask("line_hop", (32, 32), 0.1, np.random.default_rng(0))
    y = np.full((32, 32), 0.37)
    assert np.allclose(biharmonic_inpaint(y * m, m), 0.37)
