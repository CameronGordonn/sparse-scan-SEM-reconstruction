"""Train the mask-conditioned U-Net across sampling patterns, fractions and doses.

    python scripts/train_unet.py --config configs/unet.yaml
    python scripts/train_unet.py --config configs/smoke.yaml --set train.steps=50

Every training sample draws a fresh pattern (uniform over PATTERNS), a
fraction f ~ U[frac_min, frac_max], a dose log-uniform in [dose_min,
dose_max] and a dose regime, so a single model covers the whole grid.
Validation is PSNR on a frozen subset of data/eval_val.npz with masks and
noise seeded by `semrecon.data.eval_seed` (same convention as evaluation).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from .data import PatchDataset, eval_seed, load_eval_set
from .forward import acquire, pixel_dose
from .metrics import psnr
from .models.unet import UNet, count_params, reconstruct_batch
from .patterns import PATTERNS, make_mask

DEFAULTS = {
    "out_dir": "checkpoints/unet",
    "data": {
        "root": "data/nffa",
        "splits": "data/splits.csv",
        "val_set": "data/eval_val.npz",
        "patch": 256,
        "samples_per_image": 8,
        "num_workers": 4,
        "cache": False,
    },
    "sampling": {
        "patterns": list(PATTERNS),
        "frac_min": 0.05,
        "frac_max": 0.30,
        "dose_min": 5.0,
        "dose_max": 100.0,
        "fixed_dose_prob": 0.3,
    },
    "model": {"base": 32, "mults": [1, 2, 4, 8, 16], "residual": True, "groups": 8},
    "train": {
        "batch_size": 16,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "steps": 60000,
        "warmup": 1000,
        "grad_clip": 1.0,
        "ssim_weight": 0.2,
        "amp": True,
        "val_every": 2000,
        "log_every": 100,
        "val_images": 20,
        "val_fracs": [0.05, 0.1, 0.2, 0.3],
        "val_dose": 20.0,
        "seed": 0,
    },
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class SparseSampleDataset(Dataset):
    """Wraps PatchDataset; returns (x, y, mask) with a random acquisition."""

    def __init__(self, patches: PatchDataset, sampling: dict):
        self.patches = patches
        self.s = sampling

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        x = self.patches[idx]  # (1, P, P) float tensor
        xn = x[0].numpy()
        pattern = self.s["patterns"][rng.integers(len(self.s["patterns"]))]
        frac = rng.uniform(self.s["frac_min"], self.s["frac_max"])
        dose = math.exp(rng.uniform(math.log(self.s["dose_min"]), math.log(self.s["dose_max"])))
        regime = "fixed_dose" if rng.random() < self.s["fixed_dose_prob"] else "fixed_dwell"
        mask = make_mask(pattern, xn.shape, frac, rng)
        y = acquire(xn, mask, pixel_dose(regime, dose, frac), rng)
        return x, torch.from_numpy(y)[None], torch.from_numpy(mask.astype(np.float32))[None]


def build_val(cfg: dict):
    """Frozen validation inputs: list of (pattern, frac, x, y, mask) numpy arrays."""
    ev = load_eval_set(cfg["data"]["val_set"])
    t = cfg["train"]
    items = []
    for i, x in enumerate(ev["images"][: t["val_images"]]):
        for pattern in cfg["sampling"]["patterns"]:
            for frac in t["val_fracs"]:
                rng = np.random.default_rng(eval_seed(i, pattern, frac))
                mask = make_mask(pattern, x.shape, frac, rng)
                y = acquire(x, mask, t["val_dose"], rng)
                items.append((pattern, frac, x, y, mask))
    return items


@torch.no_grad()
def validate(model, items, device, batch: int = 4) -> dict:
    model.eval()
    scores: dict[str, list[float]] = {}
    for k in range(0, len(items), batch):
        chunk = items[k : k + batch]
        y = torch.from_numpy(np.stack([c[3] for c in chunk]))[:, None].to(device)
        m = torch.from_numpy(np.stack([c[4] for c in chunk]).astype(np.float32))[:, None].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = reconstruct_batch(model, y, m).float().cpu().numpy()
        for c, o in zip(chunk, out):
            scores.setdefault(c[0], []).append(psnr(o[0], c[2]))
    model.train()
    res = {f"psnr_{p}": float(np.mean(v)) for p, v in scores.items()}
    res["psnr"] = float(np.mean([np.mean(v) for v in scores.values()]))
    return res


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------

def _gauss_window(size: int = 11, sigma: float = 1.5, device=None, dtype=None):
    t = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim_torch(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM over a batch of (B,1,H,W) images, Gaussian window, valid conv."""
    w = _gauss_window(device=x.device, dtype=x.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mx**2
    syy = F.conv2d(y * y, w) - my**2
    sxy = F.conv2d(x * y, w) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return s.mean()


def loss_fn(pred, x, ssim_weight: float):
    pred = pred.float()
    return F.l1_loss(pred, x) + ssim_weight * (1 - ssim_torch(pred, x))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_sets(cfg: dict, sets: list[str]) -> dict:
    for s in sets:
        key, val = s.split("=", 1)
        d = cfg
        *path, last = key.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(val)
    return cfg


def load_config(path: Path | None, sets: list[str]) -> dict:
    user = yaml.safe_load(open(path)) if path else {}
    return apply_sets(merge(DEFAULTS, user or {}), sets)


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def infinite(loader):
    while True:
        yield from loader


def main(args):
    cfg = load_config(args.config, args.set)
    if args.out_dir:
        cfg["out_dir"] = str(args.out_dir)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t, d = cfg["train"], cfg["data"]
    torch.manual_seed(t["seed"])

    model = UNet(**cfg["model"]).to(device)
    print(f"UNet params: {count_params(model) / 1e6:.2f}M  device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, t["warmup"], t["steps"]))
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best, history = 0, -float("inf"), []
    last_ckpt = out / "last.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        step, best, history = ck["step"], ck["best_val"], ck.get("history", [])
        print(f"resumed from step {step} (best val PSNR {best:.2f})")

    def save(path: Path):
        torch.save(
            {
                "model": model.state_dict(),
                "model_cfg": model.config,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best_val": best,
                "history": history,
                "config": cfg,
            },
            path,
        )

    patches = PatchDataset(d["root"], d["splits"], "train", d["patch"], d["samples_per_image"], d["cache"])
    ds = SparseSampleDataset(patches, cfg["sampling"])
    loader = DataLoader(
        ds,
        batch_size=t["batch_size"],
        shuffle=True,
        num_workers=d["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=d["num_workers"] > 0,
    )
    val_items = build_val(cfg)
    print(f"train patches/epoch: {len(ds)}  val items: {len(val_items)}")

    model.train()
    it = infinite(loader)
    run_loss, t0 = 0.0, time.time()
    while step < t["steps"]:
        x, y, m = (v.to(device, non_blocking=True) for v in next(it))
        with torch.autocast(device.type, enabled=use_amp):
            pred = model(y, m)
        loss = loss_fn(pred, x, t["ssim_weight"])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        run_loss += loss.item()

        if step % t["log_every"] == 0:
            print(f"step {step:6d}  loss {run_loss / t['log_every']:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time() - t0) / t['log_every']:.3f}s/step", flush=True)
            run_loss, t0 = 0.0, time.time()

        if step % t["val_every"] == 0 or step == t["steps"]:
            v = validate(model, val_items, device)
            history.append({"step": step, **v})
            print(f"  val @ {step}: " + "  ".join(f"{k}={val:.2f}" for k, val in v.items()), flush=True)
            if v["psnr"] > best:
                best = v["psnr"]
                save(out / "best.pt")
            save(last_ckpt)
            (out / "history.json").write_text(json.dumps(history, indent=1))

    print(f"done. best val PSNR {best:.2f} -> {out / 'best.pt'}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--set", action="append", default=[], help="override, e.g. train.steps=100 (repeatable)")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
