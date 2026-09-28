"""Train the conditional diffusion model across patterns, fractions and doses.

    python scripts/train_diffusion.py --config configs/diffusion.yaml
    python scripts/train_diffusion.py --config configs/diffusion_smoke.yaml

Uses the same on-the-fly acquisition as the U-Net (train_unet.SparseSampleDataset):
every sample draws a fresh pattern, fraction, dose and dose regime, and masks
are per example. Training minimises the epsilon-MSE and keeps an EMA of the
weights, which is what gets sampled and saved.

Validation runs the sampler (few steps, one sample) on a frozen subset of
data/eval_val.npz. When training finishes, the best checkpoint's EMA model is
scored under each consistency mode (none / repaint / hard). The winner is
stored in best.pt as the default sampler, with the scores in consistency.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import PatchDataset
from .metrics import psnr
from .models.diffusion import (
    CONSISTENCY,
    DEFAULT_SAMPLER,
    CosineNoiseSchedule,
    DiffusionUNet,
    count_params,
    diffusion_loss,
    reconstruct_batch,
)
from .patterns import PATTERNS
from .train_unet import SparseSampleDataset, apply_sets, build_val, infinite, lr_lambda, merge

DEFAULTS = {
    "out_dir": "checkpoints/diffusion",
    "data": {
        "root": "data/nffa",
        "splits": "data/splits.csv",
        "val_set": "data/eval_val.npz",
        "patch": 128,
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
    "model": {"base": 64, "mults": [1, 2, 4, 4], "time_dim": 128, "groups": 8},
    "schedule": {"num_steps": 1000, "s": 0.008, "alpha_bar_min": 1.0e-5},
    "sampler": dict(DEFAULT_SAMPLER),
    "train": {
        "batch_size": 32,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "steps": 150000,
        "warmup": 1000,
        "grad_clip": 1.0,
        "ema": 0.999,
        "amp": True,
        "val_every": 5000,
        "log_every": 200,
        "val_images": 6,
        "val_fracs": [0.05, 0.1, 0.2, 0.3],
        "val_dose": 20.0,
        "val_crop": 256,
        "val_steps": 50,
        "val_batch": 12,
        "select_consistency": True,
        "select_samples": 2,
        "seed": 0,
    },
}


def load_config(path: Path | None, sets: list[str]) -> dict:
    user = yaml.safe_load(open(path)) if path else {}
    return apply_sets(merge(DEFAULTS, user or {}), sets)


def crop_items(items, size: int | None):
    """Top-left crop of the frozen val items, to keep validation sampling affordable."""
    if not size:
        return items
    return [(p, f, x[:size, :size], y[:size, :size], m[:size, :size]) for p, f, x, y, m in items]


@torch.no_grad()
def validate(model, items, device, sampler: dict, batch: int = 12, n_samples: int = 1, seed: int = 0) -> dict:
    model.eval()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    scores: dict[str, list[float]] = {}
    for k in range(0, len(items), batch):
        chunk = items[k : k + batch]
        y = torch.from_numpy(np.stack([c[3] for c in chunk]))[:, None].to(device)
        m = torch.from_numpy(np.stack([c[4] for c in chunk]).astype(np.float32))[:, None].to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            mean, _ = reconstruct_batch(model, y, m, n_samples, generator=g, **sampler)
        for c, o in zip(chunk, mean.float().cpu().numpy()):
            scores.setdefault(c[0], []).append(psnr(o[0], c[2]))
    res = {f"psnr_{p}": float(np.mean(v)) for p, v in scores.items()}
    res["psnr"] = float(np.mean([np.mean(v) for v in scores.values()]))
    return res


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach(), 1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def main(args):
    cfg = load_config(args.config, args.set)
    if args.out_dir:
        cfg["out_dir"] = str(args.out_dir)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t, d = cfg["train"], cfg["data"]
    torch.manual_seed(t["seed"])

    model = DiffusionUNet(**cfg["model"]).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    schedule = CosineNoiseSchedule(**cfg["schedule"]).to(device)
    ema.schedule = schedule
    print(f"DiffusionUNet params: {count_params(model) / 1e6:.2f}M  device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, t["warmup"], t["steps"]))
    use_amp = bool(t["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best, history = 0, -float("inf"), []
    last_ckpt = out / "last.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        step, best, history = ck["step"], ck["best_val"], ck.get("history", [])
        print(f"resumed from step {step} (best val PSNR {best:.2f})")

    def save(path: Path, sampler: dict | None = None):
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "model_cfg": model.config,
                "schedule_cfg": cfg["schedule"],
                "sampler": sampler or cfg["sampler"],
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
    val_items = crop_items(build_val(cfg), t["val_crop"])
    val_sampler = {**cfg["sampler"], "steps": t["val_steps"]}
    print(f"train patches/epoch: {len(ds)}  val items: {len(val_items)}")

    model.train()
    it = infinite(loader)
    run_loss, t0 = 0.0, time.time()
    while step < t["steps"]:
        x, y, m = (v.to(device, non_blocking=True) for v in next(it))
        with torch.autocast(device.type, enabled=use_amp):
            loss = diffusion_loss(model, schedule, x, y, m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema_update(ema, model, t["ema"] if step >= t["warmup"] else 0.0)
        step += 1
        run_loss += loss.item()

        if step % t["log_every"] == 0:
            print(f"step {step:6d}  loss {run_loss / t['log_every']:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time() - t0) / t['log_every']:.3f}s/step", flush=True)
            run_loss, t0 = 0.0, time.time()

        if step % t["val_every"] == 0 or step == t["steps"]:
            v = validate(ema, val_items, device, val_sampler, t["val_batch"])
            history.append({"step": step, **v})
            print(f"  val @ {step}: " + "  ".join(f"{k}={val:.2f}" for k, val in v.items()), flush=True)
            if v["psnr"] > best:
                best = v["psnr"]
                save(out / "best.pt")
            save(last_ckpt)
            (out / "history.json").write_text(json.dumps(history, indent=1))

    print(f"done. best val PSNR {best:.2f} -> {out / 'best.pt'}")
    if t["select_consistency"] and (out / "best.pt").exists():
        select_consistency(out / "best.pt", val_items, device, val_sampler, t)


def select_consistency(ckpt_path: Path, val_items, device, val_sampler: dict, t: dict) -> str:
    """Score each consistency mode on val with the best EMA model; store the winner as its default."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DiffusionUNet(**ck["model_cfg"]).to(device)
    model.load_state_dict(ck["ema"])
    model.schedule = CosineNoiseSchedule(**ck["schedule_cfg"]).to(device)
    scores = {}
    for mode in CONSISTENCY:
        v = validate(model, val_items, device, {**val_sampler, "consistency": mode}, t["val_batch"],
                     n_samples=t["select_samples"])
        scores[mode] = v
        print(f"  consistency={mode:8s} " + "  ".join(f"{k}={val:.2f}" for k, val in v.items()), flush=True)
    winner = max(scores, key=lambda k: scores[k]["psnr"])
    ck["sampler"] = {**ck["sampler"], "consistency": winner}
    torch.save(ck, ckpt_path)
    (ckpt_path.parent / "consistency.json").write_text(json.dumps({"selected": winner, "scores": scores}, indent=1))
    print(f"selected consistency={winner} -> stored as default sampler in {ckpt_path}")
    return winner


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
