"""Conditional diffusion model for sparse-scan SEM reconstruction.

Ported from ../diffusion-sparse-reconstruction-hpc (scripts/04_diffusion_model_v2.py,
scripts/04_diffusion_train.py), where it reconstructed ERA5 fields from sparse
points. What carries over:
- the (x_t, t, y, M) interface, with the condition concatenated as input channels
  (here 1 + 1 + 1 = 3 channels);
- the cosine schedule and epsilon-prediction objective;
- the two sampler guards from docs/guides/SAMPLER_FIX.md: a floor on alpha-bar
  (cos^2(pi/2) = 0 at t = T gave a 1/sqrt(alpha) ~ 100 first step that blew up
  the reverse process) and clipping the predicted x0.

What changes:
- Index convention. alpha_bar[t] = acp[t + 1] is used in both training and
  sampling. The old code trained with acp[t] but sampled with acp[t + 1], an
  off-by-one, and trained t = 0 at alpha_bar = 1, where the epsilon target is
  unlearnable.
- The time embedding is added in every residual block, not only at the
  bottleneck.
- Masks are drawn per example, and one model is trained across all patterns,
  fractions and doses. The old repo used one model per sparsity with one mask
  per batch.
- Measurement consistency during sampling is configurable:
    none     pure conditional sampling; the network sees y and M but x is never
             overwritten. Suits Poisson-noisy y, which should not be copied
             through.
    repaint  at each step, observed pixels are replaced by a sample of
             q(x_t' | y) = N(sqrt(abar') y, (1 - abar') I) at the *next* noise
             level (RePaint, Lugmayr et al. 2022). This fixes the old sampler's
             train/test mismatch of pasting clean y into a noisy state. Optional
             resampling jumps (`resample` > 1) re-noise and repeat each step.
    hard     the old repo's behaviour: paste clean y at every step. Kept for
             comparison.
- DDIM-style strided sampling (Song et al. 2021): `steps` < T uses a
  subsequence of timesteps, and eta = 1 with steps = T is ancestral DDPM.

Data are mapped from [0, 1] to [-1, 1]. The y channel is 2y - 1 at observed
pixels and 0 elsewhere, with the mask telling the two apart.

Inference size: the network is fully convolutional (no attention), so a model
trained on 128^2 patches is applied directly to 512^2 crops, padded to a multiple
of `model.factor`. GroupNorm statistics then cover a larger field of view than in
training. That is a mild shift for a conv-only net, and tiling would introduce
seams across the ensemble, so it isn't used.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONSISTENCY = ("none", "repaint", "hard")


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        ang = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)


def _groups(ch: int, groups: int) -> int:
    g = min(groups, ch)
    while ch % g:
        g -= 1
    return g


class ResBlock(nn.Module):
    """GN-SiLU-conv, add the time embedding, GN-SiLU-conv, plus a residual path."""

    def __init__(self, cin: int, cout: int, tdim: int, groups: int = 8):
        super().__init__()
        self.n1 = nn.GroupNorm(_groups(cin, groups), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(_groups(cout, groups), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class DiffusionUNet(nn.Module):
    """Epsilon-predicting U-Net with inputs [x_t, y, M] and time t.

    One ResBlock per level and additive skip connections, as in the original
    v2 model, but with time injected everywhere. Its downsampling factor is
    2 ** (len(mults) - 1).
    """

    def __init__(self, base: int = 64, mults=(1, 2, 4, 4), time_dim: int = 128, groups: int = 8,
                 in_channels: int = 3):
        super().__init__()
        self.config = {"base": base, "mults": list(mults), "time_dim": time_dim, "groups": groups,
                       "in_channels": in_channels}
        self.factor = 2 ** (len(mults) - 1)
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        chs = [base * m for m in mults]
        self.inc = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, c in enumerate(chs):
            self.enc.append(ResBlock(chs[i - 1] if i else chs[0], c, time_dim, groups))
            if i < len(chs) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
        self.mid = ResBlock(chs[-1], chs[-1], time_dim, groups)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(chs) - 2, -1, -1):
            self.up.append(nn.ConvTranspose2d(chs[i + 1], chs[i], 4, stride=2, padding=1))
            self.dec.append(ResBlock(chs[i], chs[i], time_dim, groups))
        self.out = nn.Sequential(nn.GroupNorm(_groups(chs[0], groups), chs[0]), nn.SiLU(),
                                 nn.Conv2d(chs[0], 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x_t, t, y, mask):
        temb = self.time_embed(t)
        h = self.inc(torch.cat([x_t, y, mask], dim=1))
        skips = []
        for i, blk in enumerate(self.enc):
            h = blk(h, temb)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        h = self.mid(h, temb)
        for up, blk in zip(self.up, self.dec):
            h = up(h) + skips.pop()
            h = blk(h, temb)
        return self.out(h)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------

class CosineNoiseSchedule:
    """Cosine schedule (Nichol & Dhariwal 2021) with a floor on alpha-bar.

    `abar[t]` for t = 0..T-1 is the cumulative signal level of x_t, taken as
    acp[t + 1] so that t = 0 is already slightly noisy, and floored at
    `alpha_bar_min` so that the t = T-1 endpoint is not exactly 0 (see
    SAMPLER_FIX.md). `abar_prev(t)` is the level of the previous timestep, 1 for
    t < 0.
    """

    def __init__(self, num_steps: int = 1000, s: float = 0.008, alpha_bar_min: float = 1e-5):
        self.num_steps = num_steps
        steps = torch.arange(num_steps + 1, dtype=torch.float64)
        acp = torch.cos((steps / num_steps + s) / (1 + s) * math.pi * 0.5) ** 2
        acp = acp / acp[0]
        self.abar = acp[1:].clamp_min(alpha_bar_min)

    def to(self, device) -> "CosineNoiseSchedule":
        self.abar = self.abar.to(device)
        return self

    def level(self, t: torch.Tensor) -> torch.Tensor:
        """alpha-bar at integer timesteps t, with t < 0 meaning clean (1.0)."""
        a = self.abar.to(t.device)[t.clamp_min(0)]
        return torch.where(t < 0, torch.ones_like(a), a)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.level(t).view(-1, 1, 1, 1).to(x0.dtype)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise

    def sample_timestep(self, batch: int, device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch,), device=device)


# --------------------------------------------------------------------------
# conditioning helpers
# --------------------------------------------------------------------------

def to_model_space(y: torch.Tensor, mask: torch.Tensor):
    """[0,1] measurement -> y channel in [-1,1] at observed pixels, 0 elsewhere."""
    m = mask.to(y.dtype)
    return (2 * y - 1) * m, m


def diffusion_loss(model, schedule: CosineNoiseSchedule, x: torch.Tensor, y: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """Epsilon-MSE. x, y in [0,1]; mask float (B,1,H,W)."""
    x0 = 2 * x - 1
    yc, m = to_model_space(y, mask)
    t = schedule.sample_timestep(x0.shape[0], x0.device)
    x_t, eps = schedule.add_noise(x0, t)
    return F.mse_loss(model(x_t, t, yc, m).float(), eps.float())


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------

@torch.no_grad()
def sample(
    model: DiffusionUNet,
    schedule: CosineNoiseSchedule,
    y: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 100,
    eta: float = 1.0,
    consistency: str = "none",
    resample: int = 1,
    x0_clip: float | None = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Strided DDIM/DDPM sampler. y, mask are (B,1,H,W) in [0,1]; returns x0 in [0,1].

    For each pair (t -> t') in the timestep subsequence:
        eps   = model(x_t, t, y, M)
        x0    = clip((x_t - sqrt(1-a) eps) / sqrt(a))          # x0 guard
        eps   = (x_t - sqrt(a) x0) / sqrt(1-a)                  # consistent with clipped x0
        sigma = eta * sqrt((1-a')/(1-a)) * sqrt(1 - a/a')
        x_t'  = sqrt(a') x0 + sqrt(1-a'-sigma^2) eps + sigma z
    With eta = 1 and steps = T this is exactly the DDPM posterior step.
    """
    if consistency not in CONSISTENCY:
        raise ValueError(f"consistency must be one of {CONSISTENCY}")
    device = y.device
    yc, m = to_model_space(y, mask)
    y0 = 2 * y - 1  # observed values in model space
    T = schedule.num_steps
    ts = torch.linspace(T - 1, 0, steps, device=device).round().long().unique_consecutive().tolist()
    ts_next = ts[1:] + [-1]

    def randn(shape):
        return torch.randn(shape, device=device, generator=generator)

    x = randn(y.shape)
    for t, tn in zip(ts, ts_next):
        tt = torch.full((y.shape[0],), t, device=device, dtype=torch.long)
        a = schedule.level(tt)[0].item()
        an = 1.0 if tn < 0 else schedule.level(torch.tensor([tn], device=device))[0].item()
        for r in range(resample):
            eps = model(x, tt, yc, m).float()
            x0 = (x - math.sqrt(1 - a) * eps) / math.sqrt(a)
            if x0_clip is not None:
                x0 = x0.clamp(-x0_clip, x0_clip)
            eps = (x - math.sqrt(a) * x0) / math.sqrt(1 - a)
            sigma = eta * math.sqrt(max((1 - an) / (1 - a) * (1 - a / an), 0.0))
            x_next = math.sqrt(an) * x0 + math.sqrt(max(1 - an - sigma**2, 0.0)) * eps
            if sigma > 0:
                x_next = x_next + sigma * randn(x.shape)
            if consistency == "repaint":
                known = math.sqrt(an) * y0 + math.sqrt(1 - an) * randn(x.shape)
                x_next = m * known + (1 - m) * x_next
            elif consistency == "hard":
                x_next = m * y0 + (1 - m) * x_next
            if r < resample - 1 and tn >= 0:
                # RePaint jump: diffuse x_{t'} back to level t and redo the step
                x = math.sqrt(a / an) * x_next + math.sqrt(1 - a / an) * randn(x.shape)
            else:
                x = x_next
    return ((x + 1) / 2).clamp(0, 1)


# --------------------------------------------------------------------------
# inference API
# --------------------------------------------------------------------------

DEFAULT_SAMPLER = {"steps": 100, "eta": 1.0, "consistency": "none", "resample": 1, "x0_clip": 1.0}


def load_model(ckpt_path: str | Path, device: str | torch.device = "cpu") -> DiffusionUNet:
    """Load EMA weights if present. The schedule and sampler settings are attached to the model."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DiffusionUNet(**ck["model_cfg"]).to(device)
    model.load_state_dict(ck.get("ema") or ck["model"])
    model.eval()
    model.schedule = CosineNoiseSchedule(**ck.get("schedule_cfg", {})).to(device)
    model.sampler = {**DEFAULT_SAMPLER, **ck.get("sampler", {})}
    return model


@torch.no_grad()
def reconstruct_batch(model: DiffusionUNet, y: torch.Tensor, mask: torch.Tensor, n_samples: int = 4,
                      generator: torch.Generator | None = None, **sampler_kw):
    """(B,1,H,W) -> (mean, std) over n_samples, each (B,1,H,W). Pads to a multiple of model.factor."""
    H, W = y.shape[-2:]
    f = model.factor
    ph, pw = (-H) % f, (-W) % f
    y = F.pad(y, (0, pw, 0, ph))
    mask = F.pad(mask.to(y.dtype), (0, pw, 0, ph))  # padding counts as unobserved
    B = y.shape[0]
    kw = {**getattr(model, "sampler", DEFAULT_SAMPLER), **sampler_kw}
    ys = y.repeat_interleave(n_samples, 0)
    ms = mask.repeat_interleave(n_samples, 0)
    xs = sample(model, model.schedule, ys, ms, generator=generator, **kw)
    xs = xs[..., :H, :W].view(B, n_samples, 1, H, W)
    return xs.mean(1), xs.std(1) if n_samples > 1 else torch.zeros_like(xs[:, 0])


def reconstruct(model: DiffusionUNet, y: np.ndarray, mask: np.ndarray, device: str | torch.device = "cpu",
                n_samples: int = 4, return_std: bool = False, seed: int | None = 0, **sampler_kw):
    """Single image: y (H,W) zero-filled measurement, mask (H,W) bool -> ensemble mean (H,W) in [0,1]."""
    yt = torch.from_numpy(np.asarray(y, np.float32))[None, None].to(device)
    mt = torch.from_numpy(np.asarray(mask, np.float32))[None, None].to(device)
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    mean, std = reconstruct_batch(model, yt, mt, n_samples, generator=g, **sampler_kw)
    mean = mean[0, 0].float().cpu().numpy()
    if return_std:
        return mean, std[0, 0].float().cpu().numpy()
    return mean
