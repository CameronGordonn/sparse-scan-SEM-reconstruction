"""Poisson shot-noise forward model for sparse SEM acquisition.

The clean image x in [0, 1] is read as the mean detected secondary-electron
yield per pixel, relative to the brightest signal. With beam current giving
Phi detected electrons per second at x = 1 and dwell time tau, a measured
pixel records

    n ~ Poisson(x * D),   D = Phi * tau   (expected counts at x = 1)

and the normalised measurement is y = n / D, so E[y] = x and
Var[y] = x / D. Unmeasured pixels are set to 0 and flagged by the mask.

Dose regimes
------------
fixed_dwell   D is the same for every sampling fraction; sparser scans are
              faster but each pixel is equally noisy.
fixed_dose    the total electron dose of a full raster at D_full is spent on
              the sampled pixels, so D = D_full / frac. A sparse scan then
              trades coverage for per-pixel SNR at equal specimen dose.
"""

from __future__ import annotations

import numpy as np

REGIMES = ("fixed_dwell", "fixed_dose")


def pixel_dose(regime: str, dose: float, frac: float) -> float:
    """Expected counts at x = 1 for one measured pixel."""
    if regime == "fixed_dwell":
        return dose
    if regime == "fixed_dose":
        return dose / frac
    raise ValueError(f"unknown dose regime {regime!r}")


def acquire(
    x: np.ndarray,
    mask: np.ndarray,
    dose: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a sparse, shot-noise-limited acquisition.

    x     clean image in [0, 1]
    mask  boolean, True where the beam dwells
    dose  expected counts per measured pixel at x = 1 (D above);
          np.inf gives a noiseless measurement
    """
    y = np.zeros_like(x, dtype=np.float32)
    if np.isinf(dose):
        y[mask] = x[mask]
        return y
    lam = np.clip(x[mask], 0.0, None) * dose
    y[mask] = rng.poisson(lam) / dose
    return y


def acquire_torch(x, mask, dose, generator=None):
    """Batched torch version for on-the-fly training data.

    x     (B, 1, H, W) float in [0, 1]
    mask  (B, 1, H, W) float/bool
    dose  (B,) tensor of per-sample doses
    """
    import torch

    d = dose.view(-1, 1, 1, 1).to(x)
    counts = torch.poisson(x.clamp_min(0) * d, generator=generator)
    return (counts / d) * mask.to(x)
