"""Sampling patterns for sparse SEM acquisition.

Convention: images are (H, W); rows are scan lines and the fast-scan
direction is along columns (left to right). A mask is a boolean array with
True where the beam dwells and a pixel is measured.

Patterns
--------
uniform         iid pixels drawn without replacement (the generic inpainting
                baseline; physically it needs a beam jump for almost every
                pixel).
partial_raster  a subset of full scan lines; "jittered" places one line at a
                random offset inside each of n equal strata, "random" picks
                lines uniformly without replacement.
line_hop        every line is scanned, but the beam only dwells on random-
                length segments and hops forward over the gaps between them.
                Jumps are always forward along the line, so the scan coils
                never reverse mid-line.

All generators hit the requested fraction exactly (up to rounding to an
integer pixel count) so that methods are compared at matched budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PATTERNS = ("uniform", "partial_raster", "line_hop")


def uniform(shape: tuple[int, int], frac: float, rng: np.random.Generator) -> np.ndarray:
    H, W = shape
    n = int(round(frac * H * W))
    idx = rng.choice(H * W, size=n, replace=False)
    mask = np.zeros(H * W, dtype=bool)
    mask[idx] = True
    return mask.reshape(H, W)


def partial_raster(
    shape: tuple[int, int],
    frac: float,
    rng: np.random.Generator,
    mode: str = "jittered",
) -> np.ndarray:
    H, W = shape
    n = max(1, int(round(frac * H)))
    if mode == "jittered":
        edges = np.linspace(0, H, n + 1)
        lo = np.floor(edges[:-1]).astype(int)
        hi = np.maximum(np.floor(edges[1:]).astype(int), lo + 1)
        rows = lo + (rng.random(n) * (hi - lo)).astype(int)
    elif mode == "random":
        rows = rng.choice(H, size=n, replace=False)
    else:
        raise ValueError(f"unknown partial_raster mode {mode!r}")
    mask = np.zeros((H, W), dtype=bool)
    mask[rows] = True
    return mask


def _composition(total: int, parts: int, rng: np.random.Generator, min_part: int) -> np.ndarray:
    """Random split of `total` into `parts` integers, each >= min_part."""
    free = total - parts * min_part
    if parts == 1:
        return np.array([total])
    if free < 0:
        raise ValueError("composition infeasible")
    # stars and bars: choose parts-1 bar positions among free+parts-1 slots
    bars = np.sort(rng.choice(free + parts - 1, size=parts - 1, replace=False))
    sizes = np.diff(np.concatenate(([-1], bars, [free + parts - 1]))) - 1
    return sizes + min_part


def line_hop(
    shape: tuple[int, int],
    frac: float,
    rng: np.random.Generator,
    mean_segment: float = 16.0,
) -> np.ndarray:
    """Each line gets round(frac*W) pixels split into ~k/mean_segment runs."""
    H, W = shape
    k = int(round(frac * W))
    mask = np.zeros((H, W), dtype=bool)
    if k == 0:
        return mask
    if k >= W:
        mask[:] = True
        return mask
    # number of segments, limited so interior gaps can be >= 1 pixel
    nseg = int(np.clip(round(k / mean_segment), 1, W - k + 1))
    for r in range(H):
        seg = _composition(k, nseg, rng, min_part=1)
        # nseg+1 gaps: leading and trailing may be empty, interior >= 1
        gaps = _composition(W - k + 2, nseg + 1, rng, min_part=1)
        gaps[0] -= 1
        gaps[-1] -= 1
        pos = 0
        for g, s in zip(gaps[:-1], seg):
            pos += g
            mask[r, pos : pos + s] = True
            pos += s
    return mask


def make_mask(
    pattern: str, shape: tuple[int, int], frac: float, rng: np.random.Generator, **kw
) -> np.ndarray:
    if pattern == "uniform":
        return uniform(shape, frac, rng)
    if pattern == "partial_raster":
        return partial_raster(shape, frac, rng, **kw)
    if pattern == "line_hop":
        return line_hop(shape, frac, rng, **kw)
    raise ValueError(f"unknown pattern {pattern!r}")


@dataclass(frozen=True)
class ScanTiming:
    """Scan-time cost model (seconds).

    dwell     time the beam sits on a measured pixel
    settle    coil settling time after a blanked in-line jump
    flyback   line flyback + settle before a new line starts
    """

    dwell: float = 1e-6
    settle: float = 5e-6
    flyback: float = 50e-6


def scan_stats(mask: np.ndarray) -> dict[str, int]:
    """Count pixels, scanned lines and in-line jumps in raster traversal order."""
    m = mask.astype(np.int8)
    starts = np.diff(np.pad(m, ((0, 0), (1, 0))), axis=1) == 1  # run starts per row
    runs_per_row = starts.sum(axis=1)
    lines = int((runs_per_row > 0).sum())
    return {
        "pixels": int(m.sum()),
        "lines": lines,
        "jumps": int(runs_per_row.sum()) - lines,
    }


def scan_time(mask: np.ndarray, timing: ScanTiming = ScanTiming(), dwell: float | None = None) -> float:
    s = scan_stats(mask)
    tau = timing.dwell if dwell is None else dwell
    return s["pixels"] * tau + s["jumps"] * timing.settle + s["lines"] * timing.flyback
