#!/usr/bin/env bash
# Train and evaluate every learned model on one Colab GPU runtime, unattended.
#
#   nohup bash colab/launch_all.sh > /content/launch_all.log 2>&1 &
#
# Assumes the repo at /content/repo with data/nffa, data/splits.csv and the
# frozen data/eval_{test,val}.npz in place, and the training array cache at
# $CACHE (scripts/cache_images.py). Runs, in two parallel lanes on the GPU:
#   lane A: U-Net (all patterns), then U-Net (uniform-only ablation), then their evals
#   lane B: diffusion, then its eval
# Outputs go to $OUT (Drive if mounted, else /content/runs), resumable per run.
set -u
cd /content/repo
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
CACHE=${CACHE:-/content/cache/train_uint8.npy}
if [ -d /content/drive/MyDrive ]; then OUT=${OUT:-/content/drive/MyDrive/sem-recon}; else OUT=${OUT:-/content/runs}; fi
mkdir -p "$OUT/checkpoints" "$OUT/results" "$OUT/logs"
echo "OUT=$OUT  CACHE=$CACHE  $(date)"
COMMON="--set data.cache=$CACHE --set data.num_workers=5"

lane_a() {
  python scripts/train_unet.py --config configs/unet.yaml --out-dir "$OUT/checkpoints/unet" $COMMON \
    > "$OUT/logs/unet.log" 2>&1 && echo "unet trained $(date)"
  python scripts/train_unet.py --config configs/unet.yaml --out-dir "$OUT/checkpoints/unet_uniform_only" $COMMON \
    --set "sampling.patterns=[uniform]" --set train.steps=30000 \
    > "$OUT/logs/unet_uniform_only.log" 2>&1 && echo "unet_uniform_only trained $(date)"
  for run in unet unet_uniform_only; do
    rm -f "$OUT/results/metrics_$run.csv"
    python scripts/evaluate.py --stage eval --methods unet --unet-ckpt "$OUT/checkpoints/$run/best.pt" \
      --method-name "$run" --device cuda --out "$OUT/results/metrics_$run.csv" \
      > "$OUT/logs/eval_$run.log" 2>&1 && echo "$run evaluated $(date)"
  done
}

lane_b() {
  python scripts/train_diffusion.py --config configs/diffusion.yaml --out-dir "$OUT/checkpoints/diffusion" $COMMON \
    > "$OUT/logs/diffusion.log" 2>&1 && echo "diffusion trained $(date)"
  rm -f "$OUT/results/metrics_diffusion.csv"
  python scripts/evaluate.py --stage eval --methods diffusion --diffusion-ckpt "$OUT/checkpoints/diffusion/best.pt" \
    --diffusion-samples 4 --n-images 100 --device cuda --out "$OUT/results/metrics_diffusion.csv" \
    > "$OUT/logs/eval_diffusion.log" 2>&1 && echo "diffusion evaluated $(date)"
}

lane_a & A=$!
lane_b & B=$!
wait $A $B
echo "ALL DONE $(date)"
