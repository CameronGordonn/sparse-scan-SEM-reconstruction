#!/usr/bin/env bash
# Assemble the GitHub Pages site in _site/: the write-up page, its figures (straight from
# results/figures, so they never drift from the committed results) and the PDF.
# Preview locally with: bash site/build.sh && python -m http.server -d _site
# The PDF is the same page printed (print styles hide the site chrome). After editing the page:
#   bash site/build.sh && chrome --headless --no-pdf-header-footer --virtual-time-budget=15000 \
#     --print-to-pdf=docs/results-explained.pdf _site/index.html
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf _site
mkdir -p _site/fig
cp site/index.html _site/
cp docs/results-explained.pdf _site/
# every figure the page references, failing loudly if one is missing
for f in $(grep -o 'src="fig/[^"]*"' site/index.html | sed 's/src="fig\///; s/"$//' | sort -u); do
  cp "results/figures/$f" _site/fig/
done
# the social preview image (og:image)
cp results/figures/reconstruct_example.png _site/fig/
touch _site/.nojekyll
echo "site -> _site/ ($(ls _site/fig | wc -l) figures)"
