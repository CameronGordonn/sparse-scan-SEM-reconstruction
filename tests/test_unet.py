import numpy as np
import pytest
import torch

from semrecon.models.unet import UNet, load_model, normalized_conv_fill, reconstruct
from semrecon.train_unet import loss_fn, ssim_torch


@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("shape", [(64, 64), (50, 77), (33, 16)])
def test_reconstruct_any_shape(residual, shape):
    torch.manual_seed(0)
    model = UNet(base=8, residual=residual).eval()
    rng = np.random.default_rng(0)
    mask = rng.random(shape) < 0.2
    y = np.where(mask, rng.random(shape), 0).astype(np.float32)
    out = reconstruct(model, y, mask)
    assert out.shape == shape and out.dtype == np.float32
    assert out.min() >= 0 and out.max() <= 1


def test_residual_starts_at_fill():
    model = UNet(base=8, residual=True).eval()
    y = torch.rand(1, 1, 32, 32)
    m = (torch.rand(1, 1, 32, 32) < 0.3).float()
    with torch.no_grad():
        assert torch.allclose(model(y, m), normalized_conv_fill(y * m, m))


def test_fill_is_exact_on_constant():
    m = (torch.rand(1, 1, 48, 48) < 0.1).float()
    y = torch.full_like(m, 0.37) * m
    assert torch.allclose(normalized_conv_fill(y, m), torch.full_like(m, 0.37), atol=1e-5)


def test_ssim_identity_and_ordering():
    x = torch.rand(2, 1, 40, 40)
    assert ssim_torch(x, x).item() == pytest.approx(1.0, abs=1e-6)
    assert ssim_torch(x, x + 0.05 * torch.randn_like(x)) > ssim_torch(x, x + 0.3 * torch.randn_like(x))


def test_overfit_one_batch():
    torch.manual_seed(0)
    model = UNet(base=8, residual=True)
    x = torch.rand(2, 1, 32, 32)
    m = (torch.rand_like(x) < 0.3).float()
    y = x * m
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(30):
        loss = loss_fn(model(y, m), x, 0.2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.7 * losses[0]


def test_checkpoint_roundtrip(tmp_path):
    model = UNet(base=8)
    torch.save({"model": model.state_dict(), "model_cfg": model.config}, tmp_path / "m.pt")
    loaded = load_model(tmp_path / "m.pt")
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
