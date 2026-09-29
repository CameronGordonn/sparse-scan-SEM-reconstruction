"""Diagnostics for sampled reconstructions: uncertainty quality and texture.

Uncertainty (does the per-pixel spread of the samples predict the error of their mean?)
  spearman            rank correlation between predicted std and actual |error|
  calibration_bins    RMSE of the mean within bins of predicted std; a calibrated model
                      has RMSE ≈ std in every bin
  sparsification      RMSE of the pixels kept after dropping the most uncertain fraction q;
                      compared with the oracle (dropping the largest errors), the area
                      between the two curves (AUSE) is 0 for a perfect uncertainty ranking

Texture
  radial_power_spectrum   azimuthally averaged power spectrum; blurry reconstructions lose
                          power at high frequencies relative to the ground truth
  high_freq_ratio         that loss as one number, over the band 0.15-0.5 x Nyquist
"""

from __future__ import annotations

import numpy as np

QS = np.linspace(0.0, 0.9, 10)
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 has only trapz


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel())).astype(np.float64)
    rb = np.argsort(np.argsort(b.ravel())).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def calibration_bins(std: np.ndarray, err: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count bins of predicted std -> (mean predicted std, RMSE of the error) per bin."""
    s, e = std.ravel(), err.ravel()
    order = np.argsort(s)
    bins = np.array_split(order, n_bins)
    return (np.array([s[b].mean() for b in bins]), np.array([np.sqrt(np.mean(e[b] ** 2)) for b in bins]))


def sparsification(score: np.ndarray, err: np.ndarray, qs: np.ndarray = QS) -> np.ndarray:
    """RMSE of the pixels that remain after removing the fraction q with the highest score."""
    e2 = (err.ravel() ** 2)[np.argsort(-score.ravel())]
    n = len(e2)
    # suffix sums: mean of e2[k:] for k = q*n
    suffix = np.cumsum(e2[::-1])[::-1]
    ks = np.minimum((qs * n).astype(int), n - 1)
    return np.sqrt(suffix[ks] / (n - ks))


def ause(model_curve: np.ndarray, oracle_curve: np.ndarray, qs: np.ndarray = QS) -> float:
    """Area between the model's and the oracle's sparsification curves (trapezoid over q)."""
    return float(_trapezoid(model_curve - oracle_curve, qs))


def radial_power_spectrum(img: np.ndarray, n_bins: int = 64) -> np.ndarray:
    """Azimuthally averaged |FFT|^2 of the mean-removed image, n_bins radial bins up to Nyquist."""
    x = img.astype(np.float64) - img.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    H, W = x.shape
    yy, xx = np.indices((H, W))
    r = np.hypot((yy - H // 2) / (H / 2), (xx - W // 2) / (W / 2))  # 1.0 = Nyquist along an axis
    idx = np.minimum((r * n_bins).astype(int), n_bins)
    sums = np.bincount(idx.ravel(), p.ravel(), minlength=n_bins + 1)[:n_bins]
    counts = np.bincount(idx.ravel(), minlength=n_bins + 1)[:n_bins]
    return sums / np.maximum(counts, 1)


def high_freq_ratio(pred: np.ndarray, truth: np.ndarray, lo: float = 0.15, hi: float = 0.5) -> float:
    """Power of pred relative to truth between `lo` and `hi` x Nyquist (1 = matches truth).

    The band stops at half Nyquist because the reference images are themselves noisy JPEGs:
    above that their power is mostly shot noise and compression artefacts, not specimen texture.
    """
    sp, st = radial_power_spectrum(pred), radial_power_spectrum(truth)
    a, b = int(lo * len(sp)), int(hi * len(sp))
    return float(sp[a:b].sum() / st[a:b].sum())
