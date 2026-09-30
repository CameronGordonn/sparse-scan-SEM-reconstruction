"""Sensitivity of each method to scan-coil landing errors after blanked jumps (semrecon.coils).

    python scripts/coil_errors.py --methods tv_l2 --out results/jitter/metrics_tv_l2.csv --workers 5
    python scripts/coil_errors.py --methods unet --unet-ckpt checkpoints/unet/best.pt --out results/jitter/metrics_unet.csv
    python scripts/coil_errors.py --stage plot

Existing trained models only; nothing is retrained. Rows mirror results/metrics*.csv plus `amp` (pixels).
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

from semrecon.coils import make_case_coils
from semrecon.data import load_eval_set
from semrecon.evaluate import lambda_key, run_classical
from semrecon.metrics import all_metrics

FIELDS = ["method", "regime", "dose", "pattern", "frac", "image", "category", "amp", "tau",
          "psnr", "ssim", "psnr_unobserved", "runtime_s"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["run", "plot"], default="run")
    p.add_argument("--methods", nargs="*", default=["tv_l2"])
    p.add_argument("--eval-set", type=Path, default=Path("data/eval_test.npz"))
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--patterns", nargs="*", default=["uniform", "partial_raster", "line_hop"])
    p.add_argument("--fracs", nargs="*", type=float, default=[0.10, 0.30])
    p.add_argument("--amps", nargs="*", type=float, default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--regime", default="fixed_dwell")
    p.add_argument("--dose", type=float, default=20.0)
    p.add_argument("--lambdas", type=Path, default=Path("results/tv_lambdas.json"))
    p.add_argument("--unet-ckpt", type=Path)
    p.add_argument("--method-name", default="unet", help="label for U-Net rows")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--threads", type=int, default=3, help="torch threads for the U-Net")
    p.add_argument("--out", type=Path, default=Path("results/jitter/metrics.csv"))
    p.add_argument("--results", type=Path, default=Path("results/jitter"))
    return p.parse_args()


def _row(method, a, i, p, fr, amp, cat, xhat, x, mask, rt):
    return {"method": method, "regime": a.regime, "dose": a.dose, "pattern": p, "frac": fr, "image": i,
            "category": cat, "amp": amp, "tau": a.tau, **all_metrics(xhat, x, mask), "runtime_s": rt}


def _tv_task(task, images, cats, a, lambdas):
    method, i, p, fr, amp = task
    x = images[i]
    y, mask = make_case_coils(x, i, p, fr, a.regime, a.dose, amp, a.tau)
    lam = lambdas.get(method, {}).get(lambda_key(a.regime, p, fr))
    t = time.perf_counter()
    xhat = run_classical(method, y, mask, lam, {"max_iter": 200, "tol": 1e-4})
    return _row(method, a, i, p, fr, amp, str(cats[i]), xhat, x, mask, time.perf_counter() - t)


def run(a):
    ev = load_eval_set(a.eval_set)
    images, cats = ev["images"][: a.n_images], ev["categories"][: a.n_images]
    lambdas = json.load(open(a.lambdas))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        classical = [m for m in a.methods if m in ("biharmonic", "tv_l2", "tv_kl")]
        tasks = [(m, i, p, fr, amp) for m in classical for p in a.patterns for fr in a.fracs
                 for amp in a.amps for i in range(len(images))]
        if tasks:
            from tqdm import tqdm

            fn = partial(_tv_task, images=images, cats=cats, a=a, lambdas=lambdas)
            with Pool(a.workers) as pool:
                for row in tqdm(pool.imap_unordered(fn, tasks, chunksize=2), total=len(tasks)):
                    w.writerow(row)
                    f.flush()
        if "unet" in a.methods:
            import torch

            from semrecon.models.unet import load_model, reconstruct

            torch.set_num_threads(a.threads)
            model = load_model(a.unet_ckpt, "cpu")
            for p in a.patterns:
                for fr in a.fracs:
                    for amp in a.amps:
                        for i, x in enumerate(images):
                            y, mask = make_case_coils(x, i, p, fr, a.regime, a.dose, amp, a.tau)
                            t = time.perf_counter()
                            xhat = reconstruct(model, y, mask, "cpu")
                            w.writerow(_row(a.method_name, a, i, p, fr, amp, str(cats[i]), xhat, x, mask,
                                            time.perf_counter() - t))
                            f.flush()
                    print(f"{a.method_name} {p} {fr} done", flush=True)


def plot(a):
    from semrecon import plots

    rows = []
    for path in sorted(a.results.glob("metrics*.csv")):
        with open(path) as f:
            for r in csv.DictReader(f):
                for k in ("frac", "amp", "psnr", "ssim", "psnr_unobserved"):
                    r[k] = float(r[k])
                rows.append(r)
    plots.coil_error_figure(rows, Path("results/figures/coil_errors.png"))
    (a.results / "summary.md").write_text(plots.coil_error_table(rows))
    print((a.results / "summary.md").read_text())


if __name__ == "__main__":
    args = parse_args()
    run(args) if args.stage == "run" else plot(args)
