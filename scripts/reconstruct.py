"""Try the reconstructions on your own SEM image.

Simulate a sparse scan of a full image and compare methods (scores are printed and drawn):
    python scripts/reconstruct.py my_image.tif
    python scripts/reconstruct.py my_image.tif --pattern line_hop --frac 0.1 --methods unet tv_l2 diffusion

Reconstruct a real sparse scan: the measured image (unmeasured pixels any value) plus a mask
image whose non-zero pixels mark what was measured:
    python scripts/reconstruct.py measured.png --mask mask.png

The trained weights are downloaded from the weights-v1 GitHub release on first use and
checked against their published sha256. The U-Net takes a few seconds per image on a CPU;
diffusion takes minutes on a CPU and seconds on a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from semrecon.data import banner_top
from semrecon.evaluate import lambda_key, run_classical
from semrecon.forward import acquire, pixel_dose
from semrecon.metrics import psnr, ssim
from semrecon.patterns import PATTERNS, ScanTiming, make_mask, scan_time

RELEASE = "https://github.com/CameronGordonn/sparse-scan-SEM-reconstruction/releases/download/weights-v1"
SHA256 = {
    "unet": "bb51970ad64bf590870176538ae62aceb26a9d63bfb1ea2b43ff716e2a20581a",
    "unet_uniform_only": "852418db353311f3f7baf3a8b9a385c49b79660ee99a63984712dced4c72260e",
    "diffusion": "0817980c3d19ec06a7e6e5de9a7e7c2c1554dd8fc8823c3742fd9e070c6d2593",
}
METHODS = ("biharmonic", "tv_l2", "tv_kl", "unet", "unet_uniform_only", "diffusion")
TRAIN_WIDTH = 1024  # the models were trained on images at this width (NFFA pixel scale)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="full SEM image (simulate mode) or measured image (with --mask)")
    p.add_argument("--mask", type=Path, help="real sparse scan: non-zero pixels of this image were measured")
    p.add_argument("--pattern", choices=PATTERNS, default="partial_raster")
    p.add_argument("--frac", type=float, default=0.20, help="fraction of pixels to measure (simulate mode)")
    p.add_argument("--dose", type=float, default=20.0,
                   help="expected electrons per measured pixel at full brightness; 'inf' for no noise")
    p.add_argument("--regime", choices=["fixed_dwell", "fixed_dose"], default="fixed_dwell",
                   help="fixed_dose: --dose is per pixel of a full scan, shared among the measured pixels")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=["tv_l2", "unet"])
    p.add_argument("--max-width", type=int, default=TRAIN_WIDTH,
                   help="shrink wider images to this width (keeps the pixel scale the models saw)")
    p.add_argument("--crop-banner", action="store_true", help="remove a Zeiss-style info banner at the bottom")
    p.add_argument("--weights", type=Path, default=Path("checkpoints"), help="where the weights live / go")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("reconstruction.png"), help="comparison figure")
    p.add_argument("--save-recon", action="store_true", help="also save each reconstruction as a 16-bit PNG")
    return p.parse_args(argv)


def load_gray(path: Path, max_width: int, crop_banner: bool) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):  # 16-bit detector images: scale by the actual range
        a = np.asarray(im, dtype=np.float32)
        a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
        im = Image.fromarray((a * 255).astype(np.uint8))
    im = im.convert("L")
    if im.width > max_width:
        im = im.resize((max_width, round(im.height * max_width / im.width)), Image.Resampling.LANCZOS)
    img = np.asarray(im)
    if crop_banner:
        img = img[: max(banner_top(img) - 4, 1)]
    return img.astype(np.float32) / 255.0


def weights(name: str, root: Path) -> Path:
    path = root / name / "best.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} weights (~31 MB) ...", flush=True)
        urllib.request.urlretrieve(f"{RELEASE}/{name}.pt", path.with_suffix(".part"))
        path.with_suffix(".part").rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        print(f"note: {path} is not the weights-v1 release file (e.g. a full training checkpoint); using it")
    return path


def tv_lambda(method: str, regime: str, pattern: str | None, frac: float) -> float:
    """Validation-tuned lambda for the nearest tuned (pattern, fraction); uniform if the pattern is unknown."""
    table = Path(__file__).resolve().parent.parent / "results" / "tv_lambdas.json"
    lams = json.loads(table.read_text()).get(method, {}) if table.exists() else {}
    fracs = [0.05, 0.10, 0.15, 0.20, 0.30]
    near = min(fracs, key=lambda f: abs(f - frac))
    return lams.get(lambda_key(regime, pattern or "uniform", near), 0.05)


def reconstruct(method, y, mask, args, pattern, frac, device):
    if method in ("biharmonic", "tv_l2", "tv_kl"):
        lam = None if method == "biharmonic" else tv_lambda(method, args.regime, pattern, frac)
        return run_classical(method, y, mask, lam, {"max_iter": 200, "tol": 1e-4})
    if method == "diffusion":
        from semrecon.models import diffusion as D

        model = D.load_model(weights("diffusion", args.weights), device)
        return D.reconstruct(model, y, mask, device, n_samples=4, seed=args.seed)
    from semrecon.models import unet as U

    return U.reconstruct(U.load_model(weights(method, args.weights), device), y, mask, device)


def main(argv=None):
    args = parse_args(argv)
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_gray(args.image, args.max_width, args.crop_banner)
    full_time = scan_time(np.ones(img.shape, bool), ScanTiming())

    if args.mask:  # a real sparse scan: nothing to simulate, nothing to score against
        mask = load_gray(args.mask, args.max_width, args.crop_banner) > 0
        if mask.shape != img.shape:
            sys.exit(f"mask {mask.shape} and image {img.shape} differ in size")
        truth, y, pattern, frac = None, img * mask, None, float(mask.mean())
        print(f"real scan: {frac:.1%} of {img.shape[1]}x{img.shape[0]} pixels measured")
    else:
        rng = np.random.default_rng(args.seed)
        mask = make_mask(args.pattern, img.shape, args.frac, rng)
        y = acquire(img, mask, pixel_dose(args.regime, args.dose, args.frac), rng)
        truth, pattern, frac = img, args.pattern, args.frac
        print(f"simulated {args.pattern} scan: {mask.mean():.1%} of pixels, "
              f"{scan_time(mask, ScanTiming()) / full_time:.2f}x the time of a full raster")

    panels = ([("original", truth)] if truth is not None else []) + [("measured", y)]
    for m in args.methods:
        t = time.perf_counter()
        xhat = np.clip(reconstruct(m, y, mask, args, pattern, frac, device), 0, 1)
        dt = time.perf_counter() - t
        score = f"  PSNR {psnr(xhat, truth):.2f} dB  SSIM {ssim(xhat, truth):.3f}" if truth is not None else ""
        print(f"{m:18s} {dt:6.2f} s{score}")
        panels.append((m.replace("_", " ") + (f"\n{psnr(xhat, truth):.2f} dB" if truth is not None else ""), xhat))
        if args.save_recon:
            out = args.out.with_name(f"{args.out.stem}_{m}.png")
            Image.fromarray((xhat * 65535).round().astype(np.uint16)).save(out)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6), squeeze=False)
    for ax, (title, im) in zip(axes[0], panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
