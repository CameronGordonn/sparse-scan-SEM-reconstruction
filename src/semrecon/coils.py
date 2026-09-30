"""Scan-coil position errors after blanked jumps (a sensitivity model, not a calibrated one).

After a blanked in-line jump the deflection coils overshoot and settle. Here the beam
lands displaced along the fast-scan direction and the error decays over the segment:

    e_k = A * exp(-k / tau) + sigma * N(0, 1)        k = 0, 1, ... within the segment

- Only segments that start with an in-line jump are affected. The first segment of each
  line follows the flyback, whose settling time is assumed sufficient.
- Every in-line jump in raster order is forward, so the overshoot is positive (further
  along the line).
- The detector records the clean image at the displaced position (linear interpolation
  along the row), plus the usual Poisson noise. The value is filed under the nominal
  pixel, so a reconstruction method sees misregistered data.

What this means for the three patterns:
- partial raster: no in-line jumps, so it is unaffected;
- line-hop: only the first few pixels of each segment are shifted;
- uniform: almost every pixel starts its own segment, so almost every pixel is off by about A.

A = 0 and sigma = 0 reproduce `evaluate.make_case` bit for bit.
"""

from __future__ import annotations

import numpy as np

from .data import eval_seed
from .forward import acquire, pixel_dose
from .patterns import make_mask


def position_errors(mask: np.ndarray, amp: float, tau: float = 2.0, sigma: float = 0.0,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Fast-axis landing error (pixels) for every measured pixel; 0 elsewhere."""
    H, W = mask.shape
    err = np.zeros((H, W), dtype=np.float64)
    m = mask.astype(np.int8)
    starts = np.diff(np.pad(m, ((0, 0), (1, 0))), axis=1) == 1
    if amp != 0:
        for r in range(H):
            s = np.flatnonzero(starts[r])
            if len(s) < 2:
                continue
            ends = np.flatnonzero(np.diff(np.pad(m[r], (0, 1))) == -1) + 1  # exclusive run ends
            for a, b in zip(s[1:], ends[1:]):  # skip the first run: it follows the flyback
                err[r, a:b] = amp * np.exp(-np.arange(b - a) / tau)
    if sigma > 0:
        if rng is None:
            raise ValueError("sigma > 0 needs an rng")
        after_jump = err != 0
        err[after_jump] += sigma * rng.standard_normal(int(after_jump.sum()))
    return err


def sample_displaced(x: np.ndarray, err: np.ndarray) -> np.ndarray:
    """x sampled at column c + err[r, c] (linear interpolation, clamped to the row)."""
    moved = err != 0
    if not moved.any():
        return x
    H, W = x.shape
    rr, cc = np.nonzero(moved)
    pos = np.clip(cc + err[rr, cc], 0, W - 1)
    c0 = np.minimum(np.floor(pos).astype(int), W - 2)
    t = pos - c0
    out = x.copy()
    out[rr, cc] = (1 - t) * x[rr, c0] + t * x[rr, c0 + 1]
    return out


def error_rng(index: int, pattern: str, frac: float) -> np.random.Generator:
    """Stream for jitter, independent of the mask/noise stream of the same case."""
    ss = eval_seed(index, pattern, frac)
    return np.random.default_rng(np.random.SeedSequence(ss.entropy, spawn_key=(7,)))


def make_case_coils(x: np.ndarray, index: int, pattern: str, frac: float, regime: str, dose: float,
                    amp: float, tau: float = 2.0, sigma: float = 0.0):
    """`evaluate.make_case` with coil landing errors applied before the Poisson draw."""
    rng = np.random.default_rng(eval_seed(index, pattern, frac))
    mask = make_mask(pattern, x.shape, frac, rng)
    err = position_errors(mask, amp, tau, sigma, error_rng(index, pattern, frac))
    y = acquire(sample_displaced(x, err), mask, pixel_dose(regime, dose, frac), rng)
    return y, mask
