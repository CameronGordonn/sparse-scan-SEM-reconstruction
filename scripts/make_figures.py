"""Build all result figures and the markdown summary table.

    python scripts/make_figures.py                                  # from results/metrics*.csv
    python scripts/make_figures.py --unet-ckpt ckpt.pt --diffusion-ckpt dckpt.pt   # add learned methods to the qualitative panel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from semrecon import plots
from semrecon.baselines import tv
from semrecon.data import load_eval_set
from semrecon.evaluate import lambda_key, make_case, run_classical


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", nargs="*", type=Path, help="metrics CSVs (default: results/metrics*.csv)")
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    p.add_argument("--eval-set", type=Path, default=Path("data/eval_test.npz"))
    p.add_argument("--lambdas", type=Path, default=Path("results/tv_lambdas.json"))
    p.add_argument("--image", type=int, default=0, help="eval image for qualitative/convergence figures")
    p.add_argument("--pattern", default="line_hop")
    p.add_argument("--frac", type=float, default=0.10)
    p.add_argument("--regime", default="fixed_dwell")
    p.add_argument("--dose", type=float, default=20.0)
    p.add_argument("--table-frac", type=float, default=0.10)
    p.add_argument("--unet-ckpt", type=Path)
    p.add_argument("--diffusion-ckpt", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip", nargs="*", default=[], choices=["curves", "qualitative", "convergence"])
    return p.parse_args()


def curves(args):
    paths = args.csv or sorted(Path("results").glob("metrics*.csv"))
    if not paths:
        print("no metrics CSVs found; skipping curves")
        return
    rows = plots.load_rows(paths)
    regimes = sorted({r["regime"] for r in rows})
    md = []
    for regime in regimes:
        for metric in ("psnr", "ssim", "psnr_unobserved"):
            plots.metric_vs_frac(rows, metric, regime, args.out / f"{metric}_vs_frac_{regime}.png")
        plots.metric_vs_time(rows, "psnr", regime, args.out / f"psnr_vs_time_{regime}.png")
        for metric in ("psnr", "ssim"):
            md.append(f"### {metric.upper()} at {args.table_frac:.0%} sampling, {regime.replace('_', ' ')}\n")
            md.append(plots.summary_table(rows, regime, args.table_frac, metric) + "\n")
    (args.out.parent / "summary.md").write_text("\n".join(md))
    print(f"curves for {regimes} from {[str(p) for p in paths]}")


def qualitative(args):
    x = load_eval_set(args.eval_set)["images"][args.image]
    y, mask = make_case(x, args.image, args.pattern, args.frac, args.regime, args.dose)
    lambdas = json.load(open(args.lambdas)) if args.lambdas.exists() else {}
    key = lambda_key(args.regime, args.pattern, args.frac)
    recon = {"biharmonic": run_classical("biharmonic", y, mask)}
    for m in ("tv_l2", "tv_kl"):
        recon[m] = run_classical(m, y, mask, lambdas.get(m, {}).get(key, 0.025), {"max_iter": 200, "tol": 1e-4})
    if args.unet_ckpt:
        from semrecon.models.unet import load_model, reconstruct

        recon["unet"] = reconstruct(load_model(args.unet_ckpt, args.device), y, mask, args.device)
    if args.diffusion_ckpt:
        from semrecon.models.diffusion import load_model as load_diff, reconstruct as recon_diff

        recon["diffusion"] = recon_diff(load_diff(args.diffusion_ckpt, args.device), y, mask, args.device)
    name = f"qualitative_{args.pattern}_{int(round(args.frac * 100))}pct_{args.regime}.png"
    plots.qualitative(x, recon, y, mask, args.out / name)


def convergence(args):
    """TV-L2 on one 256^2 crop: PDHG vs ADMM (with and without the DCT preconditioner)."""
    x = load_eval_set(args.eval_set)["images"][args.image][:256, :256]
    y, mask = make_case(x, args.image, args.pattern, args.frac, args.regime, args.dose)
    lam, n = 0.025, 1000
    x0 = tv.nearest_fill(y, mask)
    ref, _ = tv.pdhg(y, mask, lam, box=False, max_iter=20000, tol=1e-12, x0=x0)
    f_star = tv.objective(ref, y, mask, lam)
    runs = {
        "PDHG (Chambolle–Pock)": tv.pdhg(y, mask, lam, box=False, max_iter=n, tol=0, x0=x0, track=True)[1],
        "ADMM + DCT-PCG (5 it)": tv.admm(y, mask, lam, rho=0.5, max_iter=n, cg_iters=5, tol=0, track=True)[1],
        "ADMM + CG (5 it)": tv.admm(y, mask, lam, rho=0.5, max_iter=n, cg_iters=5, tol=0,
                                    precondition=False, track=True)[1],
    }
    f_star = min([f_star] + [min(h.objective) for h in runs.values()])
    hist = {k: (h.objective, h.seconds) for k, h in runs.items()}
    plots.convergence(hist, f_star, args.out / "tv_convergence.png")


def main(args):
    if "curves" not in args.skip:
        curves(args)
    if "qualitative" not in args.skip:
        qualitative(args)
    if "convergence" not in args.skip:
        convergence(args)
    print(f"figures -> {args.out}")


if __name__ == "__main__":
    main(parse_args())
