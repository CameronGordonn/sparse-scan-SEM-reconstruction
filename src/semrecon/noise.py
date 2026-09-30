"""Measuring the noise in real SEM data, to check the Poisson forward model.

Photon transfer from two aligned noisy frames of the same content (for a volume with
thin sections, two adjacent slices): their half-difference d = (a - b)/sqrt(2) carries
the noise of one frame, and its variance is binned by the local mean. Shot-noise-limited
data give a straight line,

    var = g * (mean - offset)          (g: grey levels per detected electron)

and the detected electrons per pixel follow as N = (mean - offset) / g. N is the squared
SNR, so it does not depend on the detector's brightness/contrast settings: at 4x the dwell
time a Poisson source gives exactly 4x the electrons.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def photon_transfer(a: np.ndarray, b: np.ndarray, n_bins: int = 24, lo: float = 20, hi: float = 230,
                    smooth: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Two frames of the same content -> (bin mean, noise variance) per brightness bin.

    Brightness is the local mean of both frames (box `smooth`), which keeps the binning
    variable itself nearly noise-free. The variance per bin ignores values beyond 4 robust
    standard deviations, so the few pixels where the two frames genuinely differ (edges
    between slices) don't inflate it; a plain MAD would not do, because quantised grey levels
    make the median jump in steps. Bins outside [lo, hi] are dropped: 8-bit clipping
    flattens them.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    m = uniform_filter((a + b) / 2, smooth)
    d = (a - b) / np.sqrt(2)
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.digitize(m.ravel(), edges) - 1
    dd = d.ravel()
    means, variances = [], []
    for k in range(n_bins):
        sel = idx == k
        if sel.sum() < 500:
            continue
        v = dd[sel]
        centre = np.median(v)
        sigma = 1.4826 * np.median(np.abs(v - centre))
        keep = np.abs(v - centre) <= 4 * max(sigma, np.std(v) / 4)
        means.append(m.ravel()[sel].mean())
        variances.append(v[keep].var())
    return np.array(means), np.array(variances)


def fit_shot_noise(means: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    """Least-squares line var = g (mean - offset); returns g, offset, R^2."""
    A = np.vstack([means, np.ones_like(means)]).T
    (g, c), *_ = np.linalg.lstsq(A, variances, rcond=None)
    pred = A @ np.array([g, c])
    r2 = 1 - np.sum((variances - pred) ** 2) / np.sum((variances - variances.mean()) ** 2)
    return {"gain": float(g), "offset": float(-c / g), "r2": float(r2)}


def electrons(mean: float, fit: dict[str, float]) -> float:
    """Detected electrons per pixel at grey level `mean` under the fitted line."""
    return (mean - fit["offset"]) / fit["gain"]


def noise_autocorrelation(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Lag-1 correlation of the noise along the fast-scan rows and across rows (0 = white)."""
    d = (a.astype(np.float64) - b.astype(np.float64))
    d -= uniform_filter(d, 9)  # remove slow structure differences
    corr = lambda u, v: float(np.corrcoef(u.ravel(), v.ravel())[0, 1])
    return {"along_rows": corr(d[:, :-1], d[:, 1:]), "across_rows": corr(d[:-1], d[1:])}


def affine_psnr(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """PSNR (peak 1) of pred against ref after the best affine map a*pred+b on `valid` pixels.

    For references recorded with different brightness/contrast settings; the same map is
    fitted for every method, so the comparison stays fair. Returns (psnr, pearson r).
    """
    p, r = pred[valid].astype(np.float64), ref[valid].astype(np.float64)
    A = np.vstack([p, np.ones_like(p)]).T
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    mse = np.mean((A @ coef - r) ** 2)
    return float(10 * np.log10(1.0 / mse)), float(np.corrcoef(p, r)[0, 1])
