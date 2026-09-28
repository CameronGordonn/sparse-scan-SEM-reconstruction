"""NFFA-Europe SEM dataset: download, preprocessing, splits and datasets.

Dataset: "NFFA-EUROPE - 100% SEM Dataset", Aversa et al., CNR-IOM, 21,169
1024x768 SEM images in 10 categories. Licence CC-BY.
DOI 10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba

The images are JPEG and carry a Zeiss information banner near the bottom
whose top row varies between ~600 and ~680; `load_image` detects it per
image and keeps only the rows above it.
"""

from __future__ import annotations

import csv
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

B2SHARE_FILES = "https://b2share.eudat.eu/records/zja8y-53j14/files/{name}.tar?download=1"
CATEGORIES = (
    "Biological",
    "Fibres",
    "Films_Coated_Surface",
    "MEMS_devices_and_electrodes",
    "Nanowires",
    "Particles",
    "Patterned_surface",
    "Porous_Sponge",
    "Powder",
    "Tips",
)
SPLITS = ("train", "val", "test")


def download(root: Path, categories=CATEGORIES, keep_tar: bool = False) -> None:
    """Download and extract category tars into root/<Category>/*.jpg."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for cat in categories:
        if (root / cat).is_dir() and any((root / cat).iterdir()):
            print(f"{cat}: present, skipping")
            continue
        tar_path = root / f"{cat}.tar"
        if not tar_path.exists():
            print(f"{cat}: downloading")
            urllib.request.urlretrieve(B2SHARE_FILES.format(name=cat), tar_path)
        with tarfile.open(tar_path) as tf:
            tf.extractall(root, filter="data")
        if not keep_tar:
            tar_path.unlink()


def banner_top(img: np.ndarray, search_from: int = 512, white: int = 240, frac: float = 0.6) -> int:
    """First row (below search_from) that is mostly white, i.e. the info banner."""
    rows = np.nonzero((img[search_from:] > white).mean(axis=1) > frac)[0]
    return int(rows[0]) + search_from if len(rows) else img.shape[0]


def load_image(path: Path, margin: int = 4) -> np.ndarray:
    """Grayscale float32 in [0, 1] with the info banner removed."""
    img = np.asarray(Image.open(path).convert("L"))
    top = banner_top(img)
    return img[: max(top - margin, 0)].astype(np.float32) / 255.0


def make_splits(root: Path, out_csv: Path, seed: int = 0, fractions=(0.8, 0.1, 0.1)) -> list[dict]:
    """Stratified-by-category split of images, written to CSV (path, category, split)."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    rows = []
    for cat in sorted(p.name for p in root.iterdir() if p.is_dir()):
        files = sorted(str(p.relative_to(root)) for p in (root / cat).glob("*.jpg"))
        perm = rng.permutation(len(files))
        n_tr = int(round(fractions[0] * len(files)))
        n_va = int(round(fractions[1] * len(files)))
        for rank, i in enumerate(perm):
            split = "train" if rank < n_tr else "val" if rank < n_tr + n_va else "test"
            rows.append({"path": files[i], "category": cat, "split": split})
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "category", "split"])
        w.writeheader()
        w.writerows(rows)
    return rows


def read_splits(csv_path: Path, split: str | None = None) -> list[dict]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def random_crop(img: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    H, W = img.shape
    i = rng.integers(0, H - size + 1)
    j = rng.integers(0, W - size + 1)
    return img[i : i + size, j : j + size]


def build_eval_set(
    root: Path,
    splits_csv: Path,
    out_path: Path,
    split: str = "test",
    n_images: int = 200,
    crop: int = 512,
    seed: int = 1234,
) -> None:
    """Freeze one crop per image (round-robin over categories) into an .npz.

    Masks and noise are *not* stored; they are regenerated from
    `eval_seed(image_index, pattern, frac)` so every method sees identical
    inputs without storing thousands of masks.
    """
    rng = np.random.default_rng(seed)
    rows = read_splits(splits_csv, split)
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for lst in by_cat.values():
        rng.shuffle(lst)
    chosen = []
    while len(chosen) < n_images and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(chosen) < n_images:
                chosen.append(by_cat[cat].pop())
    crops = np.stack([random_crop(load_image(Path(root) / r["path"]), crop, rng) for r in chosen])
    np.savez_compressed(
        out_path,
        images=(crops * 255).round().astype(np.uint8),
        paths=np.array([r["path"] for r in chosen]),
        categories=np.array([r["category"] for r in chosen]),
    )


def load_eval_set(path: Path) -> dict:
    z = np.load(path)
    return {
        "images": z["images"].astype(np.float32) / 255.0,
        "paths": z["paths"],
        "categories": z["categories"],
    }


def eval_seed(image_index: int, pattern: str, frac: float, base: int = 2024) -> np.random.SeedSequence:
    from .patterns import PATTERNS

    return np.random.SeedSequence([base, image_index, PATTERNS.index(pattern), int(round(frac * 1000))])


class PatchDataset:
    """Random clean patches for training (torch Dataset protocol).

    Masks and noise are applied on the fly by the training loop so that
    every epoch sees fresh sampling patterns, fractions and doses.
    """

    def __init__(self, root: Path, splits_csv: Path, split: str, patch: int, samples_per_image: int = 8,
                 cache: bool = False):
        self.root = Path(root)
        self.rows = read_splits(splits_csv, split)
        self.patch = patch
        self.samples_per_image = samples_per_image
        self._cache: dict[int, np.ndarray] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.rows) * self.samples_per_image

    def _image(self, i: int) -> np.ndarray:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        img = load_image(self.root / self.rows[i]["path"])
        if self._cache is not None:
            self._cache[i] = img
        return img

    def __getitem__(self, idx: int):
        import torch

        rng = np.random.default_rng()
        img = self._image(idx // self.samples_per_image)
        p = random_crop(img, self.patch, rng)
        k = rng.integers(4)
        p = np.rot90(p, k)
        if rng.random() < 0.5:
            p = p[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(p))[None]
