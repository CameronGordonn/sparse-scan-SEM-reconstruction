import numpy as np
import pytest
import torch

from semrecon.models.diffusion import (
    CONSISTENCY,
    CosineNoiseSchedule,
    DiffusionUNet,
    diffusion_loss,
    load_model,
    reconstruct,
    sample,
)
from semrecon.patterns import make_mask


def tiny():
    torch.manual_seed(0)
    m = DiffusionUNet(base=8, mults=(1, 2, 2), time_dim=16)
    m.schedule = CosineNoiseSchedule(num_steps=50)
    return m.eval()


def test_forward_shape():
    m = tiny()
    x = torch.randn(3, 1, 32, 48)
    t = torch.tensor([0, 10, 49])
    assert m(x, t, x, torch.ones_like(x)).shape == x.shape


def test_schedule_monotone_and_floored():
    s = CosineNoiseSchedule(num_steps=1000, alpha_bar_min=1e-5)
    a = s.abar
    assert (a[1:] <= a[:-1]).all()
    assert a[0] < 1.0  # t = 0 is already slightly noisy, so its epsilon target is learnable
    assert a[-1] >= 1e-5  # the SAMPLER_FIX floor: abar_T is never exactly 0
    # first reverse step coefficient 1/sqrt(alpha_T) stays bounded
    assert 1 / np.sqrt((a[-1] / a[-2]).item()) < 2.0
    assert s.level(torch.tensor([-1]))[0] == 1.0


def test_add_noise_statistics():
    s = CosineNoiseSchedule(num_steps=1000)
    x0 = torch.full((20000, 1, 1, 1), 0.5)
    t = torch.full((20000,), 500)
    xt, _ = s.add_noise(x0, t)
    a = s.abar[500].item()
    assert abs(xt.mean().item() - np.sqrt(a) * 0.5) < 0.02
    assert abs(xt.var().item() - (1 - a)) < 0.03


@pytest.mark.parametrize("consistency", CONSISTENCY)
def test_sampler_finite_and_in_range(consistency):
    m = tiny()
    y = torch.rand(2, 1, 16, 16)
    mask = (torch.rand(2, 1, 16, 16) < 0.2).float()
    out = sample(m, m.schedule, y * mask, mask, steps=8, consistency=consistency, resample=2)
    assert torch.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_hard_consistency_keeps_observations():
    m = tiny()
    y = torch.rand(1, 1, 16, 16)
    mask = (torch.rand(1, 1, 16, 16) < 0.3).float()
    out = sample(m, m.schedule, y * mask, mask, steps=5, consistency="hard")
    assert torch.allclose(out[mask.bool()], y[mask.bool()], atol=1e-5)


def test_reconstruct_odd_size_and_std(tmp_path):
    m = tiny()
    ck = tmp_path / "d.pt"
    torch.save({"model": m.state_dict(), "ema": m.state_dict(), "model_cfg": m.config,
                "schedule_cfg": {"num_steps": 50}, "sampler": {"steps": 4}}, ck)
    m2 = load_model(ck)
    x = np.random.default_rng(0).random((37, 45)).astype(np.float32)
    mask = make_mask("line_hop", x.shape, 0.2, np.random.default_rng(0))
    mean, std = reconstruct(m2, x * mask, mask, n_samples=3, return_std=True)
    assert mean.shape == x.shape and std.shape == x.shape
    assert (std >= 0).all() and std.max() > 0
    # seeded: identical results on repeat
    assert np.allclose(mean, reconstruct(m2, x * mask, mask, n_samples=3))


def test_overfit_one_batch():
    torch.manual_seed(0)
    m = DiffusionUNet(base=8, mults=(1, 2), time_dim=16).train()
    s = CosineNoiseSchedule(num_steps=50)
    x = torch.rand(4, 1, 16, 16)
    mask = (torch.rand_like(x) < 0.3).float()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    losses = []
    for _ in range(150):
        torch.manual_seed(1)  # fixed t and noise so the batch is truly repeated
        loss = diffusion_loss(m, s, x, x * mask, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.5 * np.mean(losses[:10])
