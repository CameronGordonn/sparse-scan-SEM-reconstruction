import numpy as np
from scipy.ndimage import gaussian_filter

from semrecon.uncertainty import (QS, ause, calibration_bins, high_freq_ratio, radial_power_spectrum,
                                  sparsification, spearman)


def test_perfect_uncertainty_matches_oracle():
    rng = np.random.default_rng(0)
    err = rng.normal(size=(64, 64))
    oracle = sparsification(np.abs(err), err)
    assert np.allclose(sparsification(np.abs(err), err), oracle)
    assert ause(oracle, oracle) == 0.0
    assert np.all(np.diff(oracle) <= 1e-12)  # dropping the worst pixels never raises RMSE
    random = sparsification(rng.random(err.shape), err)
    assert ause(random, oracle) > 0.1
    assert abs(oracle[0] - np.sqrt(np.mean(err**2))) < 1e-12 and QS[0] == 0


def test_calibrated_gaussian_errors():
    rng = np.random.default_rng(1)
    std = rng.uniform(0.01, 0.2, size=200_000)
    err = rng.normal(size=std.shape) * std
    s, rmse = calibration_bins(std, err)
    assert np.allclose(s, rmse, rtol=0.05)
    assert spearman(std, np.abs(err)) > 0.2


def test_blur_loses_high_frequency_power():
    rng = np.random.default_rng(2)
    x = rng.random((128, 128))
    assert abs(high_freq_ratio(x, x) - 1.0) < 1e-12
    assert high_freq_ratio(gaussian_filter(x, 2.0), x) < 0.3
    assert radial_power_spectrum(x).shape == (64,)
