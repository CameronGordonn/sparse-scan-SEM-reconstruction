"""Download NFFA-Europe SEM, write stratified splits, and freeze the eval set.

    python scripts/prepare_data.py --root data/nffa                      # all 10 categories (~13.5 GB)
    python scripts/prepare_data.py --root data/nffa --categories Fibres Porous_Sponge --n-eval 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from semrecon import data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("data/nffa"))
    p.add_argument("--categories", nargs="*", default=list(data.CATEGORIES))
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
    p.add_argument("--eval-out", type=Path, default=Path("data/eval_test.npz"))
    p.add_argument("--val-out", type=Path, default=Path("data/eval_val.npz"))
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--n-val", type=int, default=50)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main(args):
    if not args.skip_download:
        data.download(args.root, args.categories)
    rows = data.make_splits(args.root, args.splits, seed=args.seed)
    counts = {s: sum(r["split"] == s for r in rows) for s in data.SPLITS}
    print(f"splits: {counts} -> {args.splits}")
    data.build_eval_set(args.root, args.splits, args.eval_out, "test", args.n_eval, args.crop, seed=1234)
    data.build_eval_set(args.root, args.splits, args.val_out, "val", args.n_val, args.crop, seed=4321)
    print(f"eval sets -> {args.eval_out}, {args.val_out}")


if __name__ == "__main__":
    main(parse_args())
