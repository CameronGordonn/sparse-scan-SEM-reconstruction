"""Mask-conditioned U-Net for sparse-scan SEM reconstruction.

Input channels are [y zero-filled, mask], plus a cheap normalised-convolution
fill when `residual=True`; in that case the network predicts a correction to
the fill. The network is fully convolutional with four 2x downsamplings, so
H and W must be divisible by 16; `reconstruct` pads arbitrary sizes.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# cheap fill: multiscale normalised convolution
# --------------------------------------------------------------------------

def _gauss1d(sigma: float, device, dtype) -> torch.Tensor:
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    return g / g.sum()


def _blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with zero padding, (B, 1, H, W)."""
    g = _gauss1d(sigma, x.device, x.dtype)
    r = (len(g) - 1) // 2
    x = F.conv2d(x, g.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(x, g.view(1, 1, -1, 1), padding=(r, 0))


@torch.no_grad()
def normalized_conv_fill(
    y: torch.Tensor,
    mask: torch.Tensor,
    sigmas=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    min_weight: float = 0.02,
    keep_observed: bool = False,
) -> torch.Tensor:
    """Gaussian-weighted average of observed pixels.

    Starting from the largest sigma, each pixel takes the estimate from the
    smallest sigma whose local observed-weight exceeds `min_weight`, so the
    fill stays sharp near samples and still covers wide gaps (e.g. skipped
    raster lines). With keep_observed=False the observed pixels are also
    replaced by the local average, which denoises shot noise.
    """
    m = mask.to(y.dtype)
    ym = y * m
    out = None
    for s in sorted(sigmas, reverse=True):
        w = _blur(m, s)
        est = _blur(ym, s) / w.clamp_min(1e-8)
        out = est if out is None else torch.where(w > min_weight, est, out)
    return torch.where(m > 0, y, out) if keep_observed else out


# --------------------------------------------------------------------------
# U-Net
# --------------------------------------------------------------------------

def _groups(ch: int, groups: int) -> int:
    return math.gcd(ch, groups)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(_groups(cout, groups), cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(_groups(cout, groups), cout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32, mults=(1, 2, 4, 8, 16), residual: bool = True, groups: int = 8):
        super().__init__()
        self.config = {"base": base, "mults": list(mults), "residual": residual, "groups": groups}
        self.residual = residual
        chs = [base * m for m in mults]
        cin = 3 if residual else 2
        self.downs = nn.ModuleList()
        for c in chs[:-1]:
            self.downs.append(ConvBlock(cin, c, groups))
            cin = c
        self.mid = ConvBlock(chs[-2], chs[-1], groups)
        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c_skip, c_in in zip(reversed(chs[:-1]), reversed(chs[1:])):
            self.ups.append(nn.ConvTranspose2d(c_in, c_skip, 2, stride=2))
            self.dec.append(ConvBlock(2 * c_skip, c_skip, groups))
        self.head = nn.Conv2d(chs[0], 1, 1)
        if residual:  # start from the fill, learn the correction
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.factor = 2 ** (len(chs) - 1)

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(y.dtype)
        y = y * m
        if self.residual:
            fill = normalized_conv_fill(y, m)
            h = torch.cat([y, m, fill], dim=1)
        else:
            h = torch.cat([y, m], dim=1)
        skips = []
        for blk in self.downs:
            h = blk(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        h = self.mid(h)
        for up, dec in zip(self.ups, self.dec):
            h = up(h)
            h = dec(torch.cat([h, skips.pop()], dim=1))
        out = self.head(h)
        return fill + out if self.residual else out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# inference API
# --------------------------------------------------------------------------

def load_model(ckpt_path: str | Path, device: str | torch.device = "cpu") -> UNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UNet(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def reconstruct_batch(model: UNet, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W) tensors of any size; zero-pads (as unobserved pixels) to a multiple of model.factor."""
    H, W = y.shape[-2:]
    f = model.factor
    ph, pw = (-H) % f, (-W) % f
    if ph or pw:
        # pad the mask with zeros so padded pixels count as unobserved
        y = F.pad(y, (0, pw, 0, ph))
        mask = F.pad(mask.to(y.dtype), (0, pw, 0, ph))
    out = model(y, mask)
    return out[..., :H, :W].clamp(0, 1)


def reconstruct(model: UNet, y: np.ndarray, mask: np.ndarray, device: str | torch.device = "cpu") -> np.ndarray:
    """Single image: y (H,W) zero-filled measurement, mask (H,W) bool -> (H,W) in [0,1]."""
    yt = torch.from_numpy(np.asarray(y, np.float32))[None, None].to(device)
    mt = torch.from_numpy(np.asarray(mask, np.float32))[None, None].to(device)
    return reconstruct_batch(model, yt, mt)[0, 0].float().cpu().numpy()
