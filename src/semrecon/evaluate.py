"""Evaluation harness: every method sees identical seeded masks and noise.

For eval image i, pattern p and fraction f, the mask and the Poisson noise
come from one RNG seeded by `data.eval_seed(i, p, f)`, so results are
comparable across methods, runs and machines.

Rows are appended to a CSV with one line per (method, image, pattern, frac,
regime).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from .data import eval_seed, load_eval_set
from .forward import acquire, pixel_dose
from .metrics import all_metrics
from .patterns import ScanTiming, make_mask, scan_time

FIELDS = [
    "method", "regime", "dose", "pattern", "frac", "image", "category",
    "psnr", "ssim", "psnr_unobserved", "scan_time_rel", "runtime_s",
]
CLASSICAL = ("biharmonic", "tv_l2", "tv_kl")


def make_case(x: np.ndarray, index: int, pattern: str, frac: float, regime: str, dose: float):
    rng = np.random.default_rng(eval_seed(index, pattern, frac))
    mask = make_mask(pattern, x.shape, frac, rng)
    y = acquire(x, mask, pixel_dose(regime, dose, frac), rng)
    return y, mask


def lambda_key(regime: str, pattern: str, frac: float) -> str:
    return f"{regime}/{pattern}/{frac:.3f}"


def run_classical(method: str, y, mask, lam: float | None = None, tv_kw: dict | None = None):
    if method == "biharmonic":
        from .baselines.biharmonic import biharmonic_inpaint

        return biharmonic_inpaint(y, mask)
    from .baselines.tv import tv_inpaint

    data = {"tv_l2": "l2", "tv_kl": "kl"}[method]
    return tv_inpaint(y, mask, lam, data=data, **(tv_kw or {}))


def _classical_task(task, images, categories, regime, dose, lambdas, tv_kw, full_time):
    method, i, pattern, frac = task
    x = images[i]
    y, mask = make_case(x, i, pattern, frac, regime, dose)
    lam = lambdas.get(method, {}).get(lambda_key(regime, pattern, frac))
    t = time.perf_counter()
    xhat = run_classical(method, y, mask, lam, tv_kw)
    rt = time.perf_counter() - t
    return {
        "method": method, "regime": regime, "dose": dose, "pattern": pattern, "frac": frac,
        "image": i, "category": str(categories[i]),
        **all_metrics(xhat, x, mask),
        "scan_time_rel": scan_time(mask, ScanTiming()) / full_time,
        "runtime_s": rt,
    }


def evaluate(
    eval_set: Path,
    methods: list[str],
    patterns: list[str],
    fracs: list[float],
    regime: str,
    dose: float,
    out_csv: Path,
    lambdas: dict | None = None,
    tv_kw: dict | None = None,
    n_images: int | None = None,
    workers: int = 1,
    unet_ckpt: Path | None = None,
    device: str = "cpu",
    unet_name: str = "unet",
) -> None:
    ev = load_eval_set(eval_set)
    images, cats = ev["images"], ev["categories"]
    if n_images is not None:
        images, cats = images[:n_images], cats[:n_images]
    full_time = scan_time(np.ones(images.shape[1:], bool), ScanTiming())
    lambdas = lambdas or {}

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    f = open(out_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    classical = [m for m in methods if m in CLASSICAL]
    tasks = [(m, i, p, fr) for m in classical for p in patterns for fr in fracs for i in range(len(images))]
    fn = partial(_classical_task, images=images, categories=cats, regime=regime, dose=dose,
                 lambdas=lambdas, tv_kw=tv_kw, full_time=full_time)
    if tasks:
        from tqdm import tqdm

        if workers > 1:
            with Pool(workers) as pool:
                for row in tqdm(pool.imap_unordered(fn, tasks, chunksize=2), total=len(tasks)):
                    w.writerow(row)
                    f.flush()
        else:
            for row in tqdm(map(fn, tasks), total=len(tasks)):
                w.writerow(row)
                f.flush()

    if "unet" in methods:
        from .models.unet import load_model, reconstruct

        model = load_model(unet_ckpt, device)
        for p in patterns:
            for fr in fracs:
                for i, x in enumerate(images):
                    y, mask = make_case(x, i, p, fr, regime, dose)
                    t = time.perf_counter()
                    xhat = reconstruct(model, y, mask, device)
                    rt = time.perf_counter() - t
                    w.writerow({
                        "method": unet_name, "regime": regime, "dose": dose, "pattern": p, "frac": fr,
                        "image": i, "category": str(cats[i]), **all_metrics(xhat, x, mask),
                        "scan_time_rel": scan_time(mask, ScanTiming()) / full_time, "runtime_s": rt,
                    })
        f.flush()
    f.close()


def tune_lambdas(
    val_set: Path,
    methods: list[str],
    patterns: list[str],
    fracs: list[float],
    regime: str,
    dose: float,
    grid: list[float],
    n_images: int = 10,
    tv_kw: dict | None = None,
    workers: int = 1,
) -> dict:
    """Grid-search lambda per (method, regime, pattern, frac) on the val split by mean PSNR."""
    ev = load_eval_set(val_set)
    images = ev["images"][:n_images]
    jobs = [(m, p, fr, lam, i) for m in methods for p in patterns for fr in fracs for lam in grid
            for i in range(len(images))]
    fn = partial(_tune_task, images=images, regime=regime, dose=dose, tv_kw=tv_kw)
    if workers > 1:
        with Pool(workers) as pool:
            scores = pool.map(fn, jobs, chunksize=4)
    else:
        scores = list(map(fn, jobs))
    acc: dict[tuple, list[float]] = {}
    for (m, p, fr, lam, _), s in zip(jobs, scores):
        acc.setdefault((m, p, fr, lam), []).append(s)
    best: dict[str, dict[str, float]] = {}
    for m in methods:
        for p in patterns:
            for fr in fracs:
                means = {lam: np.mean(acc[(m, p, fr, lam)]) for lam in grid}
                lam_star = max(means, key=means.get)
                best.setdefault(m, {})[lambda_key(regime, p, fr)] = float(lam_star)
                edge = " (grid edge!)" if lam_star in (grid[0], grid[-1]) else ""
                print(f"{m:6s} {p:15s} f={fr:.2f}  lam*={lam_star:.4g}  psnr={means[lam_star]:.2f}{edge}")
    return best


def _tune_task(job, images, regime, dose, tv_kw):
    m, p, fr, lam, i = job
    # offset image indices so val seeds never coincide with test seeds
    y, mask = make_case(images[i], 10_000 + i, p, fr, regime, dose)
    from .metrics import psnr

    return psnr(run_classical(m, y, mask, lam, tv_kw), images[i])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tune and/or evaluate reconstruction methods.")
    p.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--stage", choices=["tune", "eval", "both"], default="both")
    p.add_argument("--methods", nargs="*")
    p.add_argument("--n-images", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--unet-ckpt", type=Path)
    p.add_argument("--method-name", default="unet", help="label for U-Net rows (e.g. unet_uniform_only)")
    p.add_argument("--eval-set", type=Path, help="override config eval_set")
    p.add_argument("--val-set", type=Path, help="override config val_set")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path)
    return p.parse_args(argv)


def main(args):
    cfg = yaml.safe_load(open(args.config))
    if args.eval_set:
        cfg["eval_set"] = str(args.eval_set)
    if args.val_set:
        cfg["val_set"] = str(args.val_set)
    methods = args.methods or cfg["methods"]
    workers = args.workers or cfg.get("workers", 1)
    out = args.out or Path(cfg["out_csv"])
    lam_path = Path(cfg["lambdas"])
    lambdas = json.load(open(lam_path)) if lam_path.exists() else {}
    for reg in cfg["regimes"]:
        regime, dose = reg["regime"], reg["dose"]
        if args.stage in ("tune", "both"):
            tv_methods = [m for m in methods if m.startswith("tv_")]
            if tv_methods:
                best = tune_lambdas(Path(cfg["val_set"]), tv_methods, cfg["patterns"], cfg["fracs"],
                                    regime, dose, cfg["lambda_grid"], cfg.get("tune_images", 10),
                                    cfg.get("tv", {}), workers)
                for m, d in best.items():
                    lambdas.setdefault(m, {}).update(d)
                lam_path.parent.mkdir(parents=True, exist_ok=True)
                json.dump(lambdas, open(lam_path, "w"), indent=1, sort_keys=True)
        if args.stage in ("eval", "both"):
            evaluate(Path(cfg["eval_set"]), methods, cfg["patterns"], cfg["fracs"], regime, dose, out,
                     lambdas, cfg.get("tv", {}), args.n_images or cfg.get("n_images"), workers,
                     args.unet_ckpt, args.device, args.method_name)


if __name__ == "__main__":
    main(parse_args())
