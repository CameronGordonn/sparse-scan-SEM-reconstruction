import numpy as np

from semrecon.coils import make_case_coils, position_errors, sample_displaced
from semrecon.evaluate import make_case
from semrecon.patterns import make_mask

SHAPE = (64, 96)


def _img():
    return np.random.default_rng(0).random(SHAPE)


def test_zero_amplitude_reproduces_make_case():
    x = _img()
    for p in ("uniform", "partial_raster", "line_hop"):
        y0, m0 = make_case(x, 3, p, 0.1, "fixed_dwell", 20)
        y1, m1 = make_case_coils(x, 3, p, 0.1, "fixed_dwell", 20, amp=0.0)
        assert np.array_equal(m0, m1) and np.array_equal(y0, y1)


def test_partial_raster_is_unaffected():
    x = _img()
    y0, _ = make_case(x, 5, "partial_raster", 0.3, "fixed_dwell", 20)
    for amp in (0.5, 2.0):
        y1, _ = make_case_coils(x, 5, "partial_raster", 0.3, "fixed_dwell", 20, amp=amp, sigma=0.3)
        assert np.array_equal(y0, y1)


def test_errors_only_after_jumps_and_decay():
    mask = np.zeros((1, 20), bool)
    mask[0, 1:4] = True      # first run: follows the flyback, no error
    mask[0, 8:14] = True     # second run: starts with a jump
    e = position_errors(mask, amp=1.0, tau=2.0)
    assert np.all(e[0, 1:4] == 0) and np.all(e[0, ~mask[0]] == 0)
    run = e[0, 8:14]
    assert run[0] == 1.0 and np.all(np.diff(run) < 0)
    assert np.allclose(run, np.exp(-np.arange(6) / 2.0))


def test_uniform_is_hit_hardest():
    rng = np.random.default_rng(1)
    frac_hit = {}
    for p in ("uniform", "line_hop", "partial_raster"):
        m = make_mask(p, SHAPE, 0.1, rng)
        frac_hit[p] = (position_errors(m, 1.0)[m] > 0.5).mean()
    assert frac_hit["uniform"] > 0.8 > frac_hit["line_hop"] and frac_hit["partial_raster"] == 0


def test_displacement_interpolates_along_row():
    x = np.tile(np.arange(10, dtype=float), (2, 1))
    err = np.zeros_like(x)
    err[0, 3] = 0.5
    err[1, 9] = 2.0          # clamped at the row end
    out = sample_displaced(x, err)
    assert out[0, 3] == 3.5 and out[1, 9] == 9.0 and out[0, 4] == 4.0
