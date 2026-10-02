"""Export the trained U-Net to ONNX for the in-browser demo on the project website.

    python scripts/export_onnx.py --out _site/models/unet.onnx

Downloads the weights-v1 release checkpoint (or uses --ckpt), exports a graph with inputs
y (1,1,H,W) zero-filled measurement and mask (1,1,H,W) in {0,1}, output x (1,1,H,W). H and W
must be multiples of 16. The normalised-convolution fill is part of the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

from semrecon.models.unet import load_model

URL = "https://github.com/CameronGordonn/sparse-scan-SEM-reconstruction/releases/download/weights-v1/unet.pt"
SHA256 = "bb51970ad64bf590870176538ae62aceb26a9d63bfb1ea2b43ff716e2a20581a"


class Clamped(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, y, mask):
        return self.model(y, mask).clamp(0, 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, help="checkpoint (default: download the weights-v1 release)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--check", action="store_true", help="compare against PyTorch with onnxruntime")
    args = p.parse_args(argv)

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = args.out.with_suffix(".pt")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, ckpt)
        if hashlib.sha256(ckpt.read_bytes()).hexdigest() != SHA256:
            sys.exit("downloaded checkpoint does not match the weights-v1 sha256")
    model = Clamped(load_model(ckpt)).eval()
    if args.ckpt is None:
        ckpt.unlink()

    rng = np.random.default_rng(0)
    y = torch.from_numpy(rng.random((1, 1, 64, 96), dtype=np.float32))
    m = torch.from_numpy((rng.random((1, 1, 64, 96)) < 0.2).astype(np.float32))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dyn = {0: "batch", 2: "height", 3: "width"}
    torch.onnx.export(model, (y * m, m), args.out, input_names=["y", "mask"], output_names=["x"],
                      dynamic_axes={"y": dyn, "mask": dyn, "x": dyn}, opset_version=17, dynamo=False)
    print(f"onnx -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.check:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        for H, W in [(64, 96), (512, 512)]:
            y = torch.from_numpy(rng.random((1, 1, H, W), dtype=np.float32))
            m = torch.from_numpy((rng.random((1, 1, H, W)) < 0.2).astype(np.float32))
            with torch.no_grad():
                ref = model(y * m, m).numpy()
            got = sess.run(None, {"y": (y * m).numpy(), "mask": m.numpy()})[0]
            print(f"{H}x{W}: max |onnx - torch| = {np.abs(got - ref).max():.2e}")


if __name__ == "__main__":
    main()
