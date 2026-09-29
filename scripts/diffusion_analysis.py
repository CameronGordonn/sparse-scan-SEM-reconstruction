"""What does the diffusion model offer beyond PSNR? Uncertainty, texture and ensemble size.

    # GPU: sample and score (resumable, chunked; --mirror copies each finished chunk, e.g. to Drive)
    python scripts/diffusion_analysis.py --stage run --out /content/da --mirror /content/drive/MyDrive/sem-recon/da \\
        --unet-ckpt checkpoints/unet/best.pt --diffusion-ckpt checkpoints/diffusion/best.pt --device cuda
    # anywhere: merge chunks, write figures and tables
    python scripts/diffusion_analysis.py --stage plot --out results/diffusion_analysis

Per test case (image, pattern, fraction) it draws --samples diffusion samples and records
  - PSNR of the ensemble mean for 1, 2, 4, ... samples, and of the U-Net
  - texture: high-frequency power relative to the ground truth, and LPIPS if installed,
    for the U-Net, the ensemble mean and a single sample
  - uncertainty: Spearman(std, |error|), calibration bins, sparsification vs oracle (AUSE)
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from semrecon.data import eval_seed, load_eval_set
from semrecon.evaluate import make_case
from semrecon.metrics import psnr, ssim
from semrecon import uncertainty as U

ENSEMBLE = (1, 2, 4, 8, 16)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["run", "plot"], required=True)
    p.add_argument("--out", type=Path, required=True, help="chunk directory (run) / figure+table directory (plot)")
    p.add_argument("--chunks", type=Path, help="plot: where the run chunks are (default: --out of run)")
    p.add_argument("--mirror", type=Path, help="run: copy each finished chunk here too; chunks found here are skipped")
    p.add_argument("--eval-set", type=Path, default=Path("data/eval_test.npz"))
    p.add_argument("--image-start", type=int, default=0)
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--chunk", type=int, default=5, help="images per chunk")
    p.add_argument("--patterns", nargs="*", default=["uniform", "partial_raster", "line_hop"])
    p.add_argument("--fracs", nargs="*", type=float, default=[0.10, 0.30])
    p.add_argument("--regime", default="fixed_dwell")
    p.add_argument("--dose", type=float, default=20.0)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--diffusion-steps", type=int, help="override the sampler's steps (e.g. for a quick smoke test)")
    p.add_argument("--example-images", nargs="*", type=int, default=[0], help="save full arrays for these images")
    p.add_argument("--unet-ckpt", type=Path)
    p.add_argument("--diffusion-ckpt", type=Path)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------- run (GPU)

def run(args):
    import torch

    from semrecon.models import diffusion as D
    from semrecon.models import unet as UN

    try:
        import lpips

        lp = lpips.LPIPS(net="alex", verbose=False).to(args.device)
    except ImportError:
        lp = None
        print("lpips not installed; skipping LPIPS")

    def lpips_score(a, b):
        if lp is None:
            return float("nan")
        t = lambda z: torch.from_numpy(np.asarray(z, np.float32))[None, None].repeat(1, 3, 1, 1).to(args.device) * 2 - 1
        with torch.no_grad():
            return float(lp(t(a), t(b)).item())

    ev = load_eval_set(args.eval_set)
    images, cats = ev["images"], ev["categories"]
    unet = UN.load_model(args.unet_ckpt, args.device)
    diff = D.load_model(args.diffusion_ckpt, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mirror:
        args.mirror.mkdir(parents=True, exist_ok=True)
    stop = min(args.image_start + args.n_images, len(images))
    for c0 in range(args.image_start, stop, args.chunk):
        name = f"chunk_{c0:03d}"
        if args.mirror and (args.mirror / f"{name}.npz").exists():
            print("skip", name)
            continue
        rows, curves, examples = [], {}, {}
        for i in range(c0, min(c0 + args.chunk, stop)):
            x = images[i]
            for p in args.patterns:
                for fr in args.fracs:
                    y, mask = make_case(x, i, p, fr, args.regime, args.dose)
                    xu = UN.reconstruct(unet, y, mask, args.device)
                    g = torch.Generator(device=args.device)
                    g.manual_seed(int(eval_seed(i, p, fr).generate_state(1)[0]))
                    yt = torch.from_numpy(y.astype(np.float32))[None, None].to(args.device)
                    mt = torch.from_numpy(mask.astype(np.float32))[None, None].to(args.device)
                    kw = {"steps": args.diffusion_steps} if args.diffusion_steps else {}
                    xs = D.sample_ensemble(diff, yt, mt, args.samples, g, **kw)[0, :, 0].float().cpu().numpy()
                    mean, std = xs.mean(0), xs.std(0, ddof=1)
                    err = mean - x
                    un = ~mask
                    key = f"{i}/{p}/{fr:.2f}"
                    row = {"image": i, "category": str(cats[i]), "pattern": p, "frac": fr,
                           "psnr_unet": psnr(xu, x), "ssim_unet": ssim(xu, x),
                           "psnr_sample": float(np.mean([psnr(s, x) for s in xs])),
                           "ssim_mean": ssim(mean, x), "ssim_sample": ssim(xs[0], x),
                           "hf_unet": U.high_freq_ratio(xu, x), "hf_mean": U.high_freq_ratio(mean, x),
                           "hf_sample": U.high_freq_ratio(xs[0], x),
                           "lpips_unet": lpips_score(xu, x), "lpips_mean": lpips_score(mean, x),
                           "lpips_sample": lpips_score(xs[0], x),
                           "spearman_std_err": U.spearman(std[un], np.abs(err[un])),
                           "rmse_mean": float(np.sqrt(np.mean(err[un] ** 2))),
                           "mean_std": float(std[un].mean())}
                    for n in ENSEMBLE:
                        if n <= args.samples:
                            row[f"psnr_mean{n}"] = psnr(xs[:n].mean(0), x)
                    model_c = U.sparsification(std[un], err[un])
                    oracle_c = U.sparsification(np.abs(err[un]), err[un])
                    rng = np.random.default_rng(0)
                    random_c = U.sparsification(rng.random(int(un.sum())), err[un])
                    row["ause"], row["ause_random"] = U.ause(model_c, oracle_c), U.ause(random_c, oracle_c)
                    cs, cr = U.calibration_bins(std[un], err[un])
                    curves[key] = np.stack([
                        np.pad(cs, (0, 64 - len(cs))), np.pad(cr, (0, 64 - len(cr))),
                        np.pad(model_c, (0, 64 - len(model_c))), np.pad(oracle_c, (0, 64 - len(oracle_c))),
                        np.pad(random_c, (0, 64 - len(random_c))),
                        U.radial_power_spectrum(x), U.radial_power_spectrum(xu),
                        U.radial_power_spectrum(mean), U.radial_power_spectrum(xs[0])])
                    if i in args.example_images:
                        examples[key] = np.stack([x, y, mask, xu, mean, xs[0], std]).astype(np.float16)
                    rows.append(row)
                    print(f"{key}: unet {row['psnr_unet']:.2f}  mean16 {row.get('psnr_mean16', float('nan')):.2f}"
                          f"  sample {row['psnr_sample']:.2f}  rho {row['spearman_std_err']:.2f}", flush=True)
        with open(args.out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed(args.out / f"{name}.npz", **{f"curves:{k}": v for k, v in curves.items()},
                            **{f"example:{k}": v for k, v in examples.items()})
        if args.mirror:  # closed local files, copied whole: a disconnect can't leave a half-written chunk
            for ext in ("csv", "npz"):
                shutil.copy(args.out / f"{name}.{ext}", args.mirror / f"{name}.{ext}.part")
                (args.mirror / f"{name}.{ext}.part").rename(args.mirror / f"{name}.{ext}")
        print("done", name, flush=True)


# ---------------------------------------------------------------- plot (CPU)

def plot(args):
    from semrecon import plots

    src = args.chunks or args.out
    rows, curves, examples = [], {}, {}
    for f in sorted(src.glob("chunk_*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append({k: (v if k in ("category", "pattern") else float(v)) for k, v in r.items()})
        z = np.load(f.with_suffix(".npz"))
        for k in z.files:
            kind, key = k.split(":", 1)
            (curves if kind == "curves" else examples)[key] = z[k]
    if not rows:
        raise SystemExit(f"no chunks in {src}")
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plots.diffusion_analysis_figures(rows, curves, examples, ENSEMBLE, Path("results/figures"))
    (args.out / "summary.md").write_text(plots.diffusion_analysis_tables(rows, ENSEMBLE))
    print(f"{len(rows)} cases -> {args.out}, figures -> results/figures")


if __name__ == "__main__":
    a = parse_args()
    run(a) if a.stage == "run" else plot(a)
