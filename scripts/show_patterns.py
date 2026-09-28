"""Figure: the three sampling patterns and their noisy measurements on one eval crop."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from semrecon.data import load_eval_set
from semrecon.forward import acquire
from semrecon.patterns import PATTERNS, ScanTiming, make_mask, scan_time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-set", type=Path, default=Path("data/eval_test.npz"))
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--frac", type=float, default=0.1)
    p.add_argument("--dose", type=float, default=20.0)
    p.add_argument("--zoom", type=int, default=128)
    p.add_argument("--out", type=Path, default=Path("results/figures/patterns.png"))
    return p.parse_args()


def main(args):
    x = load_eval_set(args.eval_set)["images"][args.index]
    rng = np.random.default_rng(0)
    full_t = scan_time(np.ones_like(x, bool), ScanTiming())
    fig, ax = plt.subplots(2, len(PATTERNS) + 1, figsize=(4 * (len(PATTERNS) + 1), 8))
    z = slice(0, args.zoom)
    ax[0, 0].imshow(x, cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_title("ground truth")
    ax[1, 0].imshow(x[z, z], cmap="gray", vmin=0, vmax=1)
    ax[1, 0].set_title(f"zoom {args.zoom}px")
    for k, p in enumerate(PATTERNS, start=1):
        m = make_mask(p, x.shape, args.frac, rng)
        y = acquire(x, m, args.dose, rng)
        t = scan_time(m, ScanTiming()) / full_t
        ax[0, k].imshow(y, cmap="gray", vmin=0, vmax=1)
        ax[0, k].set_title(f"{p}  f={m.mean():.3f}  time={t:.2f}x full")
        ax[1, k].imshow(y[z, z], cmap="gray", vmin=0, vmax=1)
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100)
    print(args.out)


if __name__ == "__main__":
    main(parse_args())
