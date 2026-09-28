import numpy as np
import pytest

from semrecon.forward import acquire, pixel_dose
from semrecon.patterns import PATTERNS, make_mask, scan_stats, scan_time, ScanTiming

SHAPE = (128, 160)


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("frac", [0.05, 0.1, 0.2, 0.3])
def test_fraction_is_hit(pattern, frac):
    m = make_mask(pattern, SHAPE, frac, np.random.default_rng(0))
    assert m.dtype == bool and m.shape == SHAPE
    assert abs(m.mean() - frac) <= 0.005 + 1.0 / SHAPE[0]  # partial_raster rounds to whole lines


def test_masks_are_seeded():
    a = make_mask("line_hop", SHAPE, 0.1, np.random.default_rng(3))
    b = make_mask("line_hop", SHAPE, 0.1, np.random.default_rng(3))
    assert (a == b).all()


def test_line_hop_structure():
    m = make_mask("line_hop", SHAPE, 0.2, np.random.default_rng(1), mean_segment=8)
    # every line is visited with the same pixel budget
    assert (m.sum(1) == m.sum(1)[0]).all()
    s = scan_stats(m)
    assert s["lines"] == SHAPE[0]
    # ~ k / mean_segment runs per line
    k = round(0.2 * SHAPE[1])
    assert s["jumps"] == SHAPE[0] * (round(k / 8) - 1)


def test_partial_raster_has_no_inline_jumps():
    m = make_mask("partial_raster", SHAPE, 0.1, np.random.default_rng(0))
    s = scan_stats(m)
    assert s["jumps"] == 0 and s["lines"] == round(0.1 * SHAPE[0])


def test_scan_time_orders_patterns():
    rng = np.random.default_rng(0)
    t = {p: scan_time(make_mask(p, SHAPE, 0.1, rng), ScanTiming()) for p in PATTERNS}
    # uniform needs a jump for nearly every pixel, raster lines need none
    assert t["partial_raster"] < t["line_hop"] < t["uniform"]


def test_poisson_moments():
    rng = np.random.default_rng(0)
    x = np.full((400, 400), 0.4, np.float32)
    mask = np.ones_like(x, bool)
    dose = 25.0
    y = acquire(x, mask, dose, rng)
    assert abs(y.mean() - 0.4) < 3e-3
    assert abs(y.var() - 0.4 / dose) < 1e-3
    # counts are integers
    assert np.allclose(y * dose, np.round(y * dose))


def test_unmeasured_pixels_are_zero():
    rng = np.random.default_rng(0)
    x = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    m = make_mask("uniform", x.shape, 0.2, rng)
    y = acquire(x, m, np.inf, rng)
    assert (y[~m] == 0).all() and np.allclose(y[m], x[m])


def test_fixed_dose_regime():
    assert pixel_dose("fixed_dwell", 10, 0.1) == 10
    assert pixel_dose("fixed_dose", 10, 0.1) == pytest.approx(100)
