import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "reconstruct.py"


def _run(*args):
    r = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_simulate_and_real_scan_modes(tmp_path):
    rng = np.random.default_rng(0)
    img = (np.clip(rng.normal(0.5, 0.15, (48, 64)), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(tmp_path / "image.png")

    out = _run(tmp_path / "image.png", "--methods", "biharmonic", "--frac", "0.3",
               "--out", tmp_path / "sim.png", "--save-recon")
    assert "PSNR" in out and (tmp_path / "sim.png").exists() and (tmp_path / "sim_biharmonic.png").exists()

    mask = np.zeros_like(img)
    mask[::4] = 255
    Image.fromarray(img * (mask > 0)).save(tmp_path / "measured.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    out = _run(tmp_path / "measured.png", "--mask", tmp_path / "mask.png", "--methods", "biharmonic",
               "--out", tmp_path / "real.png")
    assert "25.0% of 64x48 pixels measured" in out and "PSNR" not in out
