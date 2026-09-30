import numpy as np

from semrecon.noise import affine_psnr, electrons, fit_shot_noise, noise_autocorrelation, photon_transfer


def _frames(counts_at_white, gain, offset, rng, shape=(400, 400)):
    """Two Poisson frames of the same smooth scene, digitised as grey = offset + gain * counts."""
    yy, xx = np.indices(shape)
    x = 0.15 + 0.8 * (0.5 + 0.5 * np.sin(xx / 40.0) * np.cos(yy / 55.0))  # smooth content in [0.15, 0.95]
    frame = lambda: offset + gain * rng.poisson(counts_at_white * x)
    return frame(), frame(), x


def test_photon_transfer_recovers_gain_offset_and_counts():
    rng = np.random.default_rng(0)
    a, b, x = _frames(counts_at_white=40, gain=4.0, offset=10.0, rng=rng)
    fit = fit_shot_noise(*photon_transfer(a, b, lo=15, hi=170))
    assert abs(fit["gain"] - 4.0) < 0.3 and abs(fit["offset"] - 10.0) < 6 and fit["r2"] > 0.98
    mid = 10 + 4.0 * 40 * 0.5
    assert abs(electrons(mid, fit) - 20) < 2


def test_counts_scale_with_dwell_independent_of_gain():
    rng = np.random.default_rng(1)
    a1, b1, _ = _frames(10, gain=8.0, offset=0.0, rng=rng)    # short dwell, high gain
    a4, b4, _ = _frames(40, gain=2.0, offset=0.0, rng=rng)    # 4x dwell, gain turned down
    f1 = fit_shot_noise(*photon_transfer(a1, b1, lo=5, hi=75))
    f4 = fit_shot_noise(*photon_transfer(a4, b4, lo=5, hi=75))
    # compare at the same scene brightness (x = 0.5): grey = gain * counts * 0.5
    ratio = electrons(2.0 * 40 * 0.5, f4) / electrons(8.0 * 10 * 0.5, f1)
    assert abs(ratio - 4.0) < 0.5


def test_white_noise_and_affine_psnr():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(300, 300)), rng.normal(size=(300, 300))
    ac = noise_autocorrelation(a, b)
    assert abs(ac["along_rows"]) < 0.05 and abs(ac["across_rows"]) < 0.05
    ref = rng.random((64, 64))
    p, r = affine_psnr(3 * ref + 1, ref, np.ones_like(ref, bool))  # an affine copy is perfect
    assert p > 100 and r > 0.999
