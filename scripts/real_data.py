"""Real SEM data: does the noise follow the Poisson model, and do the methods hold up on real scans?

Data: Zenodo 10.5281/zenodo.20139642 (CC-BY-4.0; Chen et al., "Volumetric denoising enables
efficient acquisition of volume electron microscopy"), PFIB-SEM of mouse brain, 5 nm voxels,
Zeiss Gemini 300. Put the TIFF stacks in data/zenodo_20139642/.

    python scripts/real_data.py --stage noise    # dwell series 0.25-2 us: SNR vs dwell, noise spectrum
    python scripts/real_data.py --stage recon    # real partial raster from the 0.5 us scan vs the 2 us reference
    python scripts/real_data.py --stage budget   # equal scan time: all lines fast, or fewer lines slowly?
    python scripts/real_data.py --stage plot

noise: adjacent 5 nm slices show nearly the same structure, so their difference is noise. The
signal-to-noise power ratio (SNR) does not depend on brightness/contrast settings; for pure shot
noise it is proportional to dwell time.

recon: dropping lines from a real 0.5 us scan is physically a real partial-raster acquisition
(same detector, same noise, fewer lines). Each reconstruction is scored against the aligned
2 us reference after the best affine brightness/contrast map on unsaturated pixels (the two stacks
were recorded with different settings). A simulated twin of every case replaces the real noise
by the repo's Poisson model at the dose that gives the same noise power, to see whether the
simulation predicts real performance. The twin's clean image is the *adjacent* reference slice
(5 nm away: the same structure, independent noise), so the reference's own noise can't leak
into the twin and flatter it. Baseline: the full 0.5 us scan, raw and TV-L2-denoised (lambda
tuned against the reference on held-out slices, which favours the baseline).
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from semrecon.evaluate import lambda_key, run_classical
from semrecon.forward import acquire
from semrecon.noise import affine_psnr
from semrecon.patterns import ScanTiming, make_mask, scan_time

DATA = Path("data/zenodo_20139642")
OUT = Path("results/real_data")
DWELL = {0.25: "PFIB_5x5x5nm_0d25us.tif", 0.5: "PFIB_5x5x5nm_0d5us.tif", 0.75: "PFIB_5x5x5nm_0d75us.tif",
         1.0: "PFIB_5x5x5nm_1us.tif", 1.5: "PFIB_5x5x5nm_1d5us.tif", 2.0: "PFIB_5x5x5nm_2us.tif"}
FAST, REF = "PFIB_seg_0d5us_Raw_data_5x5x5nm.tif", "PFIB_seg_2us_Reference_5x5x5nm.tif"
FRACS = (0.10, 0.20, 0.30)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["noise", "recon", "budget", "plot"], required=True)
    p.add_argument("--slices", type=int, default=20, help="recon: number of evenly spaced slices")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--unet-ckpt", type=Path, default=Path("checkpoints/unet/best.pt"))
    p.add_argument("--workers", type=int, default=5)
    return p.parse_args()


def read_slice(path: Path, k: int) -> np.ndarray:
    im = Image.open(path)
    im.seek(k)
    return np.asarray(im, dtype=np.float64)


def snr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Adjacent slices -> (signal/noise power ratio, noise variance in grey levels^2)."""
    nv = np.var((a - b) / np.sqrt(2))
    return float(((a.var() + b.var()) / 2 - nv) / nv), float(nv)


def row_spectrum(a: np.ndarray, b: np.ndarray, bands: int = 8) -> list[float]:
    """Noise power along the fast-scan rows in `bands` frequency bands, normalised to mean 1."""
    d = a - b
    ps = np.mean(np.abs(np.fft.rfft(d - d.mean(), axis=1)) ** 2, 0)[1:]
    ps /= ps.mean()
    return [float(ps[i * len(ps) // bands:(i + 1) * len(ps) // bands].mean()) for i in range(bands)]


# ---------------------------------------------------------------- noise

def stage_noise(args):
    rows = []
    for dwell, name in DWELL.items():
        path = DATA / name
        if not path.exists():
            print("missing", path)
            continue
        n = Image.open(path).n_frames
        for k in range(0, n - 1, max((n - 1) // 30, 1)):
            a, b = read_slice(path, k)[150:-150, 150:-150], read_slice(path, k + 1)[150:-150, 150:-150]
            s, nv = snr(a, b)
            d = a - b
            rows.append({"dwell_us": dwell, "slice": k, "snr": s, "noise_var": nv, "mean": float(a.mean()),
                         "corr_along": float(np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]),
                         "corr_across": float(np.corrcoef(d[:-1].ravel(), d[1:].ravel())[0, 1]),
                         **{f"band{i}": v for i, v in enumerate(row_spectrum(a, b))}})
        print(f"{dwell} us: {n} slices, median SNR {np.median([r['snr'] for r in rows if r['dwell_us'] == dwell]):.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- recon

def _classical(job, lambdas):
    key, method, y, mask, ref, valid, fr = job
    if method == "full fast scan + TV-L2":  # fr carries the tuned lambda here
        return key, method, run_classical("tv_l2", y, mask, fr, {"max_iter": 200, "tol": 1e-4})
    lam = None if method == "biharmonic" else lambdas.get(method, {}).get(
        lambda_key("fixed_dwell", "partial_raster", fr), 0.05)
    xhat = run_classical(method, y, mask, lam, {"max_iter": 200, "tol": 1e-4})
    return key, method, xhat


def tune_full_scan_lambda(fast_p, ref_p, eval_slices, c, grid=(0.025, 0.05, 0.1, 0.2, 0.4, 0.8)):
    """TV-L2 denoising strength for the full-scan baseline, chosen against the reference on
    held-out slices (between the evaluation slices), so the baseline gets its best shot."""
    held = sorted({int(k) + 4 for k in eval_slices[::5]})
    scores = {lam: [] for lam in grid}
    for k in held:
        F, R = read_slice(fast_p, k), read_slice(ref_p, k)
        H, W = F.shape
        sl = (slice((H - c) // 2, (H - c) // 2 + c), slice((W - c) // 2, (W - c) // 2 + c))
        fast, ref = F[sl] / 255, R[sl] / 255
        valid = (R[sl] > 2) & (R[sl] < 253)
        for lam in grid:
            x = run_classical("tv_l2", fast, np.ones_like(fast, bool), lam, {"max_iter": 200, "tol": 1e-4})
            scores[lam].append(affine_psnr(x, ref, valid)[0])
    best = max(grid, key=lambda lam: np.mean(scores[lam]))
    print("full-scan TV lambda", best, {lam: round(float(np.mean(v)), 2) for lam, v in scores.items()}, flush=True)
    return best


def stage_recon(args):
    from semrecon.models.unet import load_model, reconstruct

    fast_p, ref_p = DATA / FAST, DATA / REF
    n = Image.open(fast_p).n_frames
    ks = np.linspace(0, n - 2, args.slices).round().astype(int)
    c = args.crop
    lambdas = json.loads(Path("results/tv_lambdas.json").read_text())
    unet = load_model(args.unet_ckpt, "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    full_lam = tune_full_scan_lambda(fast_p, ref_p, ks, c)
    rows, jobs, cases = [], [], {}
    for k in ks:
        F, R = read_slice(fast_p, k), read_slice(ref_p, k)
        F2, R2 = read_slice(fast_p, k + 1), read_slice(ref_p, k + 1)
        H, W = F.shape
        r0, c0 = (H - c) // 2, (W - c) // 2
        sl = (slice(r0, r0 + c), slice(c0, c0 + c))
        fast, ref, fast2, ref2 = F[sl] / 255, R[sl] / 255, F2[sl] / 255, R2[sl] / 255
        valid = (R[sl] > 2) & (R[sl] < 253)
        # simulated twin: the reference mapped onto the fast scan's brightness scale, with Poisson
        # noise at the dose that reproduces the fast scan's measured noise power
        A = np.vstack([ref2[valid], np.ones(valid.sum())]).T
        coef, *_ = np.linalg.lstsq(A, fast[valid], rcond=None)
        twin_clean = np.clip(coef[0] * ref2 + coef[1], 1e-3, 1)
        noise_var = np.var((fast - fast2) / np.sqrt(2))
        dose_eq = float(twin_clean.mean() / noise_var)
        rng = np.random.default_rng([2024, 777, int(k)])
        full = np.ones_like(fast, bool)
        rows.append({"slice": int(k), "source": "real", "method": "full fast scan", "frac": 1.0, "time": 1.0,
                     "dose_eq": dose_eq, **dict(zip(("psnr", "r"), affine_psnr(fast, ref, valid)))})
        twin_full = acquire(twin_clean, full, dose_eq, rng)
        rows.append({"slice": int(k), "source": "simulated", "method": "full fast scan", "frac": 1.0, "time": 1.0,
                     "dose_eq": dose_eq, **dict(zip(("psnr", "r"), affine_psnr(twin_full, ref, valid)))})
        for source, y in (("real", fast), ("simulated", twin_full)):
            cases[(int(k), source, 1.0)] = (ref, valid, 1.0, dose_eq)
            jobs.append(((int(k), source, 1.0), "full fast scan + TV-L2", y, full, ref, valid, full_lam))
        for fr in FRACS:
            mask = make_mask("partial_raster", fast.shape, fr, np.random.default_rng([2024, 778, int(k), int(fr * 1000)]))
            t = scan_time(mask, ScanTiming()) / scan_time(full, ScanTiming())
            for source, y in (("real", fast * mask), ("simulated", acquire(twin_clean, mask, dose_eq, rng))):
                key = (int(k), source, fr)
                cases[key] = (ref, valid, t, dose_eq)
                xhat = reconstruct(unet, y, mask, "cpu")
                rows.append({"slice": int(k), "source": source, "method": "unet", "frac": fr, "time": t,
                             "dose_eq": dose_eq, **dict(zip(("psnr", "r"), affine_psnr(xhat, ref, valid)))})
                for m in ("biharmonic", "tv_l2"):
                    jobs.append((key, m, y, mask, ref, valid, fr))
                if k == ks[len(ks) // 2] and fr == 0.20:
                    np.savez_compressed(OUT / f"example_{source}.npz", fast=fast, ref=ref, y=y, mask=mask, unet=xhat)
        print("slice", k, "dose_eq", round(dose_eq, 1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for key, m, xhat in pool.imap_unordered(partial(_classical, lambdas=lambdas), jobs, chunksize=2):
            ref, valid, t, dose_eq = cases[key]
            k, source, fr = key
            rows.append({"slice": k, "source": source, "method": m, "frac": fr, "time": t, "dose_eq": dose_eq,
                         **dict(zip(("psnr", "r"), affine_psnr(xhat, ref, valid)))})
            if k == ks[len(ks) // 2] and fr == 0.20:
                z = dict(np.load(OUT / f"example_{source}.npz"))
                z[m] = xhat
                np.savez_compressed(OUT / f"example_{source}.npz", **z)
    with open(OUT / "recon.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows ->", OUT / "recon.csv")


# ---------------------------------------------------------------- budget

# rigid offsets of each dwell volume relative to the 2 us volume (phase correlation on tiles;
# the 1.5 us volume is left out: its tile offsets vary, i.e. it is not rigidly registered)
SHIFT = {0.5: (17, 17), 1.0: (0, 0), 2.0: (0, 0)}
BUDGETS = {  # budget, in full 0.5 us scans -> options (dwell us, fraction of lines)
    1.0: [(0.5, 1.0), (1.0, 0.5), (2.0, 0.25)],
    0.5: [(0.5, 0.5), (1.0, 0.25), (2.0, 0.125)],
    0.25: [(0.5, 0.25), (1.0, 0.125), (2.0, 0.0625)],
}


def _budget_case(k, c, dwell):
    """Slice k of one dwell volume, registered to the 2 us volume, centre crop, in [0, 1]."""
    im = read_slice(DATA / DWELL[dwell], k)
    dy, dx = SHIFT[dwell]
    H, W = im.shape
    r0, c0 = (H - c) // 2 + dy, (W - c) // 2 + dx
    return im[r0:r0 + c, c0:c0 + c] / 255


def _budget_ref(k, c):
    """Mean of the 2 us slices k-1 and k+1: the structure of slice k within 5 nm, with noise
    independent of every option's data. Returns (ref, valid)."""
    r = (_budget_case(k - 1, c, 2.0) + _budget_case(k + 1, c, 2.0)) / 2
    return r, (r > 2 / 255) & (r < 253 / 255)


def _run_tv(job):
    tag, y, mask, lam = job
    return tag, run_classical("tv_l2", y, mask, lam, {"max_iter": 200, "tol": 1e-4})


def stage_budget(args):
    from semrecon.models.unet import load_model, reconstruct

    n = Image.open(DATA / DWELL[2.0]).n_frames
    c = args.crop
    ks = np.linspace(2, n - 3, args.slices).round().astype(int)
    held = (ks[:-1] + ks[1:]) // 2
    held = held[:: max(len(held) // 4, 1)][:4]
    options = sorted({o for opts in BUDGETS.values() for o in opts})
    grid = (0.0015, 0.003, 0.006, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4)

    def measure(k, dwell, fr, seed):
        x = _budget_case(k, c, dwell)
        full = np.ones_like(x, bool)
        mask = full if fr >= 1 else make_mask("partial_raster", x.shape, fr, np.random.default_rng(seed))
        return x * mask, mask

    # TV strength per option, tuned against the reference on held-out slices
    lam = {}
    with Pool(args.workers) as pool:
        jobs = []
        for k in held:
            for dwell, fr in options:
                y, mask = measure(int(k), dwell, fr, [2024, 779, int(k), int(fr * 1e4)])
                jobs += [((int(k), dwell, fr, g), y, mask, g) for g in grid]
        scores = {}
        for (k, dwell, fr, g), xhat in pool.imap_unordered(_run_tv, jobs, chunksize=2):
            ref, valid = _budget_ref(k, c)
            scores.setdefault((dwell, fr, g), []).append(affine_psnr(xhat, ref, valid)[0])
    for dwell, fr in options:
        lam[(dwell, fr)] = max(grid, key=lambda g: np.mean(scores[(dwell, fr, g)]))
        edge = " (grid edge)" if lam[(dwell, fr)] in (grid[0], grid[-1]) else ""
        print(f"TV lambda {dwell} us {fr:.4g}: {lam[(dwell, fr)]}{edge}", flush=True)

    unet = load_model(args.unet_ckpt, "cpu")
    rows, jobs, meta = [], [], {}
    for k in ks:
        k = int(k)
        ref, valid = _budget_ref(k, c)
        for dwell, fr in options:
            y, mask = measure(k, dwell, fr, [2024, 780, k, int(fr * 1e4)])
            base = {"slice": k, "dwell_us": dwell, "frac": fr, "time": dwell * fr / 0.5}
            if fr >= 1:
                rows.append({**base, "method": "raw", **dict(zip(("psnr", "r"), affine_psnr(y, ref, valid)))})
            else:
                rows.append({**base, "method": "biharmonic", **dict(zip(("psnr", "r"), affine_psnr(
                    run_classical("biharmonic", y, mask), ref, valid)))})
                rows.append({**base, "method": "unet", "outside_training": fr > 0.3,
                             **dict(zip(("psnr", "r"), affine_psnr(reconstruct(unet, y, mask, "cpu"), ref, valid)))})
            jobs.append(((k, dwell, fr), y, mask, lam[(dwell, fr)]))
            meta[(k, dwell, fr)] = (base, ref, valid)
        print("slice", k, flush=True)
    with Pool(args.workers) as pool:
        for tag, xhat in pool.imap_unordered(_run_tv, jobs, chunksize=2):
            base, ref, valid = meta[tag]
            rows.append({**base, "method": "tv_l2", **dict(zip(("psnr", "r"), affine_psnr(xhat, ref, valid)))})
    fields = ["slice", "dwell_us", "frac", "time", "method", "outside_training", "psnr", "r"]
    with open(OUT / "budget.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    (OUT / "budget_lambdas.json").write_text(json.dumps({f"{d}us/{fr}": v for (d, fr), v in lam.items()}, indent=1))
    print(len(rows), "rows ->", OUT / "budget.csv")


# ---------------------------------------------------------------- plot

def stage_plot(args):
    from semrecon import plots

    noise = list(csv.DictReader(open(OUT / "noise.csv"))) if (OUT / "noise.csv").exists() else []
    recon = list(csv.DictReader(open(OUT / "recon.csv")))
    examples = {s: dict(np.load(OUT / f"example_{s}.npz")) for s in ("real", "simulated")
                if (OUT / f"example_{s}.npz").exists()}
    md = plots.real_data_figures(noise, recon, examples, Path("results/figures"))
    if (OUT / "budget.csv").exists():
        md += "\n" + plots.real_budget_figure(list(csv.DictReader(open(OUT / "budget.csv"))), Path("results/figures"))
    (OUT / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    a = parse_args()
    {"noise": stage_noise, "recon": stage_recon, "budget": stage_budget, "plot": stage_plot}[a.stage](a)
