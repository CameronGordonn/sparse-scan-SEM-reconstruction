"""Image-quality metrics. Images are float in [0, 1]."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def psnr(pred: np.ndarray, truth: np.ndarray, region: np.ndarray | None = None) -> float:
    """PSNR in dB with data range 1; optionally restricted to a boolean region."""
    err = (np.clip(pred, 0, 1) - truth) ** 2
    mse = err[region].mean() if region is not None else err.mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(structural_similarity(np.clip(pred, 0, 1), truth, data_range=1.0))


def all_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {
        "psnr": psnr(pred, truth),
        "ssim": ssim(pred, truth),
        "psnr_unobserved": psnr(pred, truth, ~mask),
    }
