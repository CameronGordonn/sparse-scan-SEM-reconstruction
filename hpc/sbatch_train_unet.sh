#!/bin/bash
# Fallback to Colab: train the U-Net on a SLURM GPU node.
#   sbatch hpc/sbatch_train_unet.sh
#   sbatch --export=ALL,RUN=unet_uniform_only,EXTRA="--set sampling.patterns=[uniform]" hpc/sbatch_train_unet.sh
# Expects data/nffa, data/splits.csv and data/eval_val.npz under PROJECT_DIR
# (run scripts/prepare_data.py once on a login/data node).
#SBATCH -J unet_train
#SBATCH -p 96x24gpu4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -o logs/unet_train_%j.out

set -euo pipefail

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV="${SEM_CONDA_ENV:-semrecon}"
conda activate "$CONDA_ENV"

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"
mkdir -p logs
RUN="${RUN:-unet}"
CONFIG="${CONFIG:-configs/unet.yaml}"
EXTRA="${EXTRA:-}"
echo "Working dir: $(pwd)  env: $CONDA_ENV  python: $(which python)"
echo "RUN: $RUN  CONFIG: $CONFIG  EXTRA: $EXTRA"

for f in data/splits.csv data/eval_val.npz; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f (run scripts/prepare_data.py first)"
        exit 2
    fi
done

# Resumes automatically from checkpoints/$RUN/last.pt, so the job can be resubmitted after a timeout.
srun python scripts/train_unet.py \
    --config "$CONFIG" \
    --out-dir "checkpoints/$RUN" \
    --set data.num_workers="${SLURM_CPUS_PER_TASK:-4}" \
    $EXTRA
