#!/usr/bin/env bash
# Assemble the GitHub Pages site in _site/: the demo page, its scripts and examples, the PDF,
# and the U-Net exported to ONNX from the weights-v1 release (needs torch and onnx).
#   bash site/build.sh                       # export the model with $PYTHON (default: python)
#   ONNX=path/unet.onnx bash site/build.sh   # reuse an earlier export
# Preview with: python -m http.server -d _site   (the demo needs http://, not file://)
#
# The PDF write-up is docs/writeup.html printed; after editing it:
#   chrome --headless --no-pdf-header-footer --virtual-time-budget=15000 \
#     --print-to-pdf=docs/results-explained.pdf docs/writeup.html
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf _site
mkdir -p _site/fig _site/models
cp -r site/index.html site/demo.js site/tv-worker.js site/examples _site/
cp docs/results-explained.pdf _site/
cp results/figures/reconstruct_example.png _site/fig/   # the social preview image (og:image)
if [ -n "${ONNX:-}" ]; then
  cp "$ONNX" _site/models/unet.onnx
else
  "${PYTHON:-python}" scripts/export_onnx.py --out _site/models/unet.onnx
fi
touch _site/.nojekyll
echo "site -> _site/"
