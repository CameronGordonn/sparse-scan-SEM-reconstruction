"""Result figures and tables from the metrics CSV(s) written by `evaluate.py`.

Figures
-------
metric_vs_frac   PSNR / SSIM vs sampling fraction; one panel per pattern,
                 one line per method, 95% bootstrap CI over images.
metric_vs_time   PSNR vs relative scan time (settle + flyback model); one
                 panel per method, one line per pattern.
qualitative      GT, measurement and each method's reconstruction on one crop.
convergence      PDHG vs ADMM objective suboptimality for TV-L2.

Colour: fixed categorical slots (validated for CVD separation); every series
also has its own marker and a legend, so identity never rests on colour alone.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .uncertainty import QS as U_QS

SURFACE = "#fcfcfb"
INK = "#1f1f1e"
INK_MUTED = "#6b6a63"
GRID = "#e6e5e0"

# Fixed method -> slot mapping; colour follows the method in every figure.
METHOD_STYLE = {
    "biharmonic":        {"color": "#2a78d6", "marker": "o", "label": "Biharmonic"},
    "tv_l2":             {"color": "#eb6834", "marker": "s", "label": "TV-L2 (PDHG)"},
    "tv_kl":             {"color": "#1baf7a", "marker": "^", "label": "TV-Poisson (PDHG)"},
    "unet":              {"color": "#eda100", "marker": "D", "label": "U-Net"},
    "unet_uniform_only": {"color": "#e87ba4", "marker": "v", "label": "U-Net (uniform-only)"},
    "diffusion":         {"color": "#008300", "marker": "P", "label": "Diffusion"},
}
PATTERN_STYLE = {
    "uniform":        {"color": "#008300", "marker": "o", "label": "uniform"},
    "partial_raster": {"color": "#4a3aa7", "marker": "s", "label": "partial raster"},
    "line_hop":       {"color": "#e34948", "marker": "^", "label": "line-hop"},
}
PATTERN_ORDER = ["uniform", "partial_raster", "line_hop"]
METRIC_LABEL = {"psnr": "PSNR (dB)", "ssim": "SSIM", "psnr_unobserved": "PSNR on unmeasured pixels (dB)"}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "font.size": 10, "legend.frameon": False,
    })


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                for k in ("frac", "dose", "psnr", "ssim", "psnr_unobserved", "scan_time_rel", "runtime_s"):
                    r[k] = float(r[k])
                rows.append(r)
    # de-duplicate reruns: last row per (method, regime, pattern, frac, image) wins
    uniq = {(r["method"], r["regime"], r["pattern"], r["frac"], r["image"]): r for r in rows}
    return list(uniq.values())


def matched(rows) -> tuple[list[dict], int]:
    """Keep only images evaluated by every method, per (regime, pattern, frac).

    Methods run on different image subsets (e.g. a cheaper --n-images run)
    would otherwise be averaged over different, not equally hard, images.
    Returns the kept rows and the number dropped.
    """
    methods = defaultdict(set)
    images = defaultdict(set)
    for r in rows:
        k = (r["regime"], r["pattern"], r["frac"])
        methods[k].add(r["method"])
        images[(k, r["method"])].add(r["image"])
    common = {k: set.intersection(*(images[(k, m)] for m in ms)) for k, ms in methods.items()}
    kept = [r for r in rows if r["image"] in common[(r["regime"], r["pattern"], r["frac"])]]
    return kept, len(rows) - len(kept)


def _methods(rows):
    present = {r["method"] for r in rows}
    return [m for m in METHOD_STYLE if m in present] + sorted(present - METHOD_STYLE.keys())


def bootstrap_ci(v: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate(rows, metric, regime):
    """{(method, pattern): [(frac, mean, lo, hi, mean_time)]} sorted by frac."""
    g = defaultdict(list)
    for r in rows:
        if r["regime"] == regime:
            g[(r["method"], r["pattern"], r["frac"])].append((r[metric], r["scan_time_rel"]))
    out = defaultdict(list)
    for (m, p, fr), v in g.items():
        a = np.array(v)
        lo, hi = bootstrap_ci(a[:, 0])
        out[(m, p)].append((fr, a[:, 0].mean(), lo, hi, a[:, 1].mean()))
    return {k: sorted(v) for k, v in out.items()}


def metric_vs_frac(rows, metric: str, regime: str, out: Path) -> None:
    _style()
    agg = aggregate(rows, metric, regime)
    methods = _methods(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, p in zip(axes, PATTERN_ORDER):
        for m in methods:
            if (m, p) not in agg:
                continue
            s = METHOD_STYLE.get(m, {"color": INK_MUTED, "marker": "x", "label": m})
            a = np.array(agg[(m, p)])
            ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=s["color"], alpha=0.15, lw=0)
            ax.plot(a[:, 0], a[:, 1], color=s["color"], marker=s["marker"], label=s["label"],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(PATTERN_STYLE[p]["label"], loc="left", fontsize=11)
        ax.set_xlabel("sampling fraction")
        fracs = sorted({t[0] for (m, q), v in agg.items() if q == p for t in v})
        ax.set_xticks(fracs, [f"{f:.0%}" for f in fracs])
    axes[0].set_ylabel(METRIC_LABEL[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{METRIC_LABEL[metric]} vs sampling fraction — {regime.replace('_', ' ')}"
                 f" (mean, 95% bootstrap CI over images)", y=1.09, fontsize=11, color=INK)
    _save(fig, out)


def metric_vs_time(rows, metric: str, regime: str, out: Path) -> None:
    _style()
    agg = aggregate(rows, metric, regime)
    methods = _methods(rows)
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 3.8), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        for p in PATTERN_ORDER:
            if (m, p) not in agg:
                continue
            s = PATTERN_STYLE[p]
            a = np.array(agg[(m, p)])
            ax.plot(a[:, 4], a[:, 1], color=s["color"], marker=s["marker"], label=s["label"],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(METHOD_STYLE.get(m, {"label": m})["label"], loc="left", fontsize=11)
        ax.set_xlabel("scan time / full raster")
    axes[0, 0].set_ylabel(METRIC_LABEL[metric])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    _save(fig, out)


def summary_table(rows, regime: str, frac: float, metric: str = "psnr") -> str:
    """Markdown table: method x pattern at one fraction, mean ± CI half-width."""
    agg = aggregate(rows, metric, regime)
    lines = ["| method | " + " | ".join(PATTERN_STYLE[p]["label"] for p in PATTERN_ORDER) + " |",
             "|---|" + "---|" * len(PATTERN_ORDER)]
    for m in _methods(rows):
        cells = []
        for p in PATTERN_ORDER:
            hit = [t for t in agg.get((m, p), []) if abs(t[0] - frac) < 1e-9]
            cells.append(f"{hit[0][1]:.2f} ± {(hit[0][3] - hit[0][2]) / 2:.2f}" if hit else "—")
        lines.append(f"| {METHOD_STYLE.get(m, {'label': m})['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _per_image(rows, regime, method, category=None):
    """{image: mean PSNR over this regime's (pattern, frac) cases}."""
    g = defaultdict(list)
    for r in rows:
        if r["regime"] == regime and r["method"] == method and (category is None or r["category"] == category):
            g[r["image"]].append(r["psnr"])
    return {i: float(np.mean(v)) for i, v in g.items()}


def _categories(rows, regime):
    """Categories sorted by U-Net mean PSNR (hardest first), else alphabetically."""
    cats = sorted({r["category"] for r in rows if r["regime"] == regime})
    if any(r["method"] == "unet" for r in rows):
        cats.sort(key=lambda c: np.mean(list(_per_image(rows, regime, "unet", c).values())))
    return cats


def category_dots(rows, regime: str, out: Path) -> None:
    """Dot plot: mean PSNR per category (rows) and method (markers), over all patterns and fractions."""
    _style()
    cats, methods = _categories(rows, regime), _methods(rows)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(cats) + 1.4))
    for k in range(len(cats)):
        ax.axhline(k, color=GRID, lw=0.8, zorder=0)
    for m in methods:
        s = METHOD_STYLE.get(m, {"color": INK_MUTED, "marker": "x", "label": m})
        xs = [np.mean(list(_per_image(rows, regime, m, c).values())) for c in cats]
        ax.plot(xs, range(len(cats)), ls="none", color=s["color"], marker=s["marker"], markersize=8,
                label=s["label"], markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    n = len(_per_image(rows, regime, methods[0], cats[0]))
    ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("PSNR (dB), mean over all patterns and fractions")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    ax.set_title(f"By specimen category — {regime.replace('_', ' ')} ({n} test images per category)",
                 loc="left", fontsize=11)
    _save(fig, out)


def category_table(rows, regime: str, seed: int = 0) -> str:
    """Markdown: mean PSNR per category and method, plus the paired U-Net − TV-L2 gain with a 95% CI."""
    cats, methods = _categories(rows, regime), _methods(rows)
    label = lambda m: METHOD_STYLE.get(m, {"label": m})["label"]
    paired = "unet" in methods and "tv_l2" in methods
    head = ["category", "n"] + [label(m) for m in methods] + (["U-Net − TV-L2"] if paired else [])
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for c in cats:
        per = {m: _per_image(rows, regime, m, c) for m in methods}
        cells = [c.replace("_", " "), str(len(per[methods[0]]))] + [f"{np.mean(list(per[m].values())):.2f}" for m in methods]
        if paired:
            d = np.array([per["unet"][i] - per["tv_l2"][i] for i in per["unet"]])
            lo, hi = bootstrap_ci(d, seed=seed)
            cells.append(f"{d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def frontier(rows, regime: str) -> dict[str, list[tuple[float, float, str, float]]]:
    """Per method, the (scan time, PSNR, pattern, frac) points not beaten by any faster configuration."""
    agg = aggregate(rows, "psnr", regime)
    out = {}
    for m in _methods(rows):
        pts = sorted((t[4], t[1], p, t[0]) for (mm, p), v in agg.items() if mm == m for t in v)
        best, front = -np.inf, []
        for t, psnr_, p, fr in pts:
            if psnr_ > best:
                best = psnr_
                front.append((t, psnr_, p, fr))
        out[m] = front
    return out


def frontier_plot(rows, regime: str, out: Path) -> None:
    """Best PSNR reachable within a scan-time budget, per method (any pattern, any fraction)."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fronts = frontier(rows, regime)
    t_max = max(t for pts in fronts.values() for t, *_ in pts)
    for m, pts in fronts.items():
        s = METHOD_STYLE.get(m, {"color": INK_MUTED, "marker": "x", "label": m})
        a = np.array([(t, v) for t, v, _, _ in pts])
        # hold the last value out to the largest budget: a bigger budget never does worse
        line = np.vstack([a, [t_max, a[-1, 1]]])
        ax.step(line[:, 0], line[:, 1], where="post", color=s["color"], lw=2, label=s["label"])
        ax.plot(a[:, 0], a[:, 1], ls="none", color=s["color"], marker=s["marker"],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xscale("log")
    ticks = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("scan-time budget (relative to a full raster)")
    ax.set_ylabel("best PSNR reachable (dB)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    ax.set_title(f"Best configuration within a scan-time budget — {regime.replace('_', ' ')}",
                 loc="left", fontsize=11)
    _save(fig, out)


def frontier_table(rows, regime: str, budgets=(0.1, 0.2, 0.3, 0.5, 1.0)) -> str:
    """Markdown: per method, the best PSNR and the (pattern, fraction, time) achieving it within each budget.

    Budgets get 1% slack: a raster at fraction f takes f plus a little flyback, and should count as f.
    """
    short = {"uniform": "uniform", "partial_raster": "raster", "line_hop": "line-hop"}
    lines = ["| method | " + " | ".join(f"≤ {b:g}×" for b in budgets) + " |",
             "|---|" + "---|" * len(budgets)]
    for m, pts in frontier(rows, regime).items():
        cells = []
        for b in budgets:
            ok = [p for p in pts if p[0] <= 1.01 * b]
            cells.append(f"{ok[-1][1]:.2f} ({short[ok[-1][2]]} {ok[-1][3]:.0%}, {ok[-1][0]:.2f}×)" if ok else "—")
        lines.append(f"| {METHOD_STYLE.get(m, {'label': m})['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def qualitative(x, recon: dict[str, np.ndarray], y, mask, out: Path, zoom: int = 160) -> None:
    """Row 1: full crops; row 2: top-left zoom. recon maps method -> image."""
    _style()
    panels = [("ground truth", x), ("measured (zero-filled)", y)] + [
        (METHOD_STYLE.get(m, {"label": m})["label"], r) for m, r in recon.items()]
    fig, axes = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 5.4))
    for k, (title, img) in enumerate(panels):
        for row, im in enumerate((img, img[:zoom, :zoom])):
            ax = axes[row, k]
            ax.imshow(np.clip(im, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[0, k].set_title(title, fontsize=9, loc="left")
    axes[0, 0].set_ylabel(f"full {x.shape[0]}×{x.shape[1]}", fontsize=9)
    axes[1, 0].set_ylabel(f"top-left {zoom}×{zoom}", fontsize=9)
    _save(fig, out)


def convergence(histories: dict[str, tuple[list[float], list[float]]], f_star: float, out: Path) -> None:
    """histories: label -> (objective per iteration, cumulative seconds per iteration)."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    styles = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D")]
    for (label, (obj, secs)), (c, mk) in zip(histories.items(), styles):
        gap = np.maximum((np.array(obj) - f_star) / abs(f_star), 1e-12)
        it = np.arange(1, len(gap) + 1)
        every = max(len(gap) // 8, 1)
        axes[0].semilogy(it, gap, color=c, marker=mk, markevery=every, label=label,
                         markeredgecolor=SURFACE)
        axes[1].semilogy(secs, gap, color=c, marker=mk, markevery=every, markeredgecolor=SURFACE)
    axes[0].set_xlabel("iteration")
    axes[1].set_xlabel("wall time (s)")
    axes[0].set_ylabel("relative suboptimality  (F − F*) / F*")
    axes[0].legend()
    _save(fig, out)


def _save(fig, out: Path):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ diffusion analysis (scripts/diffusion_analysis.py)

SAMPLE_STYLE = {"color": "#4a3aa7", "marker": "o", "label": "Diffusion, single sample"}
MEAN_STYLE = {**METHOD_STYLE["diffusion"], "label": "Diffusion, ensemble mean"}


def diffusion_analysis_figures(rows, curves, examples, ensemble, out: Path) -> None:
    _style()
    out = Path(out)
    # 1. PSNR vs ensemble size, per pattern, with the U-Net as a dashed reference
    fig, ax = plt.subplots(figsize=(7, 4))
    ns = [n for n in ensemble if f"psnr_mean{n}" in rows[0]]
    for p in PATTERN_ORDER:
        rs = [r for r in rows if r["pattern"] == p]
        if not rs:
            continue
        s = PATTERN_STYLE[p]
        ax.plot(ns, [np.mean([r[f"psnr_mean{n}"] for r in rs]) for n in ns], color=s["color"], marker=s["marker"],
                label=f"diffusion mean, {s['label']}", markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axhline(np.mean([r["psnr_unet"] for r in rs]), color=s["color"], ls="--", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("diffusion samples averaged")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Ensemble size (dashed: U-Net, same pattern)", loc="left", fontsize=11)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    _save(fig, out / "diffusion_ensemble.png")

    # 2. uncertainty: calibration and sparsification, averaged over cases
    C = np.stack(list(curves.values()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    s_bins, rmse_bins = C[:, 0, :10].mean(0), C[:, 1, :10].mean(0)
    lim = max(s_bins.max(), rmse_bins.max()) * 1.1
    axes[0].plot([0, lim], [0, lim], color=INK_MUTED, lw=1, ls="--", label="perfectly calibrated")
    axes[0].plot(s_bins, rmse_bins, color=MEAN_STYLE["color"], marker="o", markeredgecolor=SURFACE, label="diffusion")
    axes[0].set_xlabel("predicted std (sample spread)")
    axes[0].set_ylabel("actual RMSE of the ensemble mean")
    axes[0].set_title("Calibration (10 equal-count bins)", loc="left", fontsize=11)
    axes[0].legend(loc="upper left")
    qs = U_QS
    for idx, lab, col, ls in ((2, "drop highest predicted std", MEAN_STYLE["color"], "-"),
                              (3, "oracle: drop largest errors", INK, "--"),
                              (4, "random", INK_MUTED, ":")):
        axes[1].plot(qs, C[:, idx, :10].mean(0), color=col, ls=ls, label=lab)
    axes[1].set_xlabel("fraction of unmeasured pixels removed")
    axes[1].set_ylabel("RMSE of the remaining pixels")
    axes[1].set_title("Sparsification", loc="left", fontsize=11)
    axes[1].legend(loc="lower left")
    _save(fig, out / "diffusion_uncertainty.png")

    # 3. texture: power spectrum relative to the ground truth
    fig, ax = plt.subplots(figsize=(7, 4))
    f = (np.arange(64) + 0.5) / 64
    gt = C[:, 5, :]
    for idx, s in ((6, METHOD_STYLE["unet"]), (7, MEAN_STYLE), (8, SAMPLE_STYLE)):
        ratio = np.exp(np.mean(np.log(C[:, idx, :] / gt), 0))  # geometric mean over cases
        ax.plot(f, ratio, color=s["color"], label=s["label"])
    ax.axhline(1, color=INK_MUTED, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("spatial frequency (fraction of Nyquist)")
    ax.set_ylabel("power / ground-truth power")
    ax.set_title("Texture: how much fine detail survives (1 = matches truth)", loc="left", fontsize=11)
    ax.legend(loc="lower left")
    _save(fig, out / "diffusion_texture.png")

    # 4. example panel
    if examples:
        key = next((k for k in examples if "line_hop/0.10" in k), next(iter(examples)))
        x, y, mask, xu, mean, one, std = examples[key].astype(np.float32)
        err = np.abs(mean - x)
        panels = [("ground truth", x, "gray"), ("U-Net", xu, "gray"), ("diffusion mean", mean, "gray"),
                  ("one diffusion sample", one, "gray"), ("sample spread (std)", std, "magma"),
                  ("|error| of the mean", err, "magma")]
        fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3))
        vmax = np.percentile(np.concatenate([std.ravel(), err.ravel()]), 99)
        for ax, (t, im, cm) in zip(axes, panels):
            ax.imshow(im[:192, :192], cmap=cm, vmin=0, vmax=1 if cm == "gray" else vmax, interpolation="nearest")
            ax.set_title(t, fontsize=9, loc="left")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
        img, pat, fr = key.split("/")
        fig.suptitle(f"Test image {img}, {pat.replace('_', '-')} {float(fr):.0%}, top-left 192×192",
                     fontsize=10, color=INK)
        _save(fig, out / "diffusion_example.png")


def diffusion_analysis_tables(rows, ensemble) -> str:
    nan = lambda v: np.nanmean(v) if np.isfinite(v).any() else float("nan")
    col = lambda rs, k: np.array([r[k] for r in rs], float)
    md = [f"{len(rows)} cases: {len({r['image'] for r in rows})} test images × patterns × fractions.\n",
          "### Accuracy vs texture\n",
          "| pattern | PSNR U-Net | PSNR diffusion mean | PSNR one sample | high-freq power U-Net | mean | one sample | LPIPS U-Net | mean | one sample |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in PATTERN_ORDER + ["all"]:
        rs = rows if p == "all" else [r for r in rows if r["pattern"] == p]
        if not rs:
            continue
        nm = max(n for n in ensemble if f"psnr_mean{n}" in rs[0])
        md.append(f"| {PATTERN_STYLE.get(p, {'label': 'all'})['label']} | {nan(col(rs, 'psnr_unet')):.2f} | "
                  f"{nan(col(rs, f'psnr_mean{nm}')):.2f} | {nan(col(rs, 'psnr_sample')):.2f} | "
                  f"{nan(col(rs, 'hf_unet')):.2f} | {nan(col(rs, 'hf_mean')):.2f} | {nan(col(rs, 'hf_sample')):.2f} | "
                  f"{nan(col(rs, 'lpips_unet')):.3f} | {nan(col(rs, 'lpips_mean')):.3f} | {nan(col(rs, 'lpips_sample')):.3f} |")
    md.append("\nHigh-frequency power is relative to the ground truth over 0.15–0.5 × Nyquist "
              "(1 = same amount of fine detail; above half Nyquist the reference is mostly noise). LPIPS is a learned perceptual distance (lower = closer).\n")
    md += ["### Ensemble size\n", "| samples averaged | " + " | ".join(str(n) for n in ensemble if f"psnr_mean{n}" in rows[0]) + " |",
           "|---|" + "---|" * len([n for n in ensemble if f"psnr_mean{n}" in rows[0]]),
           "| PSNR (dB) | " + " | ".join(f"{nan(col(rows, f'psnr_mean{n}')):.2f}" for n in ensemble if f"psnr_mean{n}" in rows[0]) + " |",
           "\n### Uncertainty\n",
           "| pattern | Spearman(std, abs error) | AUSE (lower is better) | AUSE, random ranking |", "|---|---|---|---|"]
    for p in PATTERN_ORDER + ["all"]:
        rs = rows if p == "all" else [r for r in rows if r["pattern"] == p]
        if rs:
            md.append(f"| {PATTERN_STYLE.get(p, {'label': 'all'})['label']} | {nan(col(rs, 'spearman_std_err')):.2f} | "
                      f"{nan(col(rs, 'ause')):.4f} | {nan(col(rs, 'ause_random')):.4f} |")
    return "\n".join(md) + "\n"
