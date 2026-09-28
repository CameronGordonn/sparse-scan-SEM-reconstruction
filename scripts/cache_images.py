"""Decode a split once into a memory-mappable uint8 array for fast training.

    python scripts/cache_images.py --out /dev/shm/train_uint8.npy
    python scripts/train_unet.py --config configs/unet.yaml --set data.cache=/dev/shm/train_uint8.npy
"""

import argparse
from pathlib import Path

from semrecon.data import build_array_cache

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--root", type=Path, default=Path("data/nffa"))
p.add_argument("--splits", type=Path, default=Path("data/splits.csv"))
p.add_argument("--split", default="train")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--workers", type=int, default=8)
a = p.parse_args()
build_array_cache(a.root, a.splits, a.split, a.out, a.workers)
print(f"cache -> {a.out}")
