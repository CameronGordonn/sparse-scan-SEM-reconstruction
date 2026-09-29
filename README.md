# Sparse-scan SEM reconstruction

[![CI](https://github.com/CameronGordonn/sparse-scan-SEM-reconstruction/actions/workflows/ci.yml/badge.svg)](https://github.com/CameronGordonn/sparse-scan-SEM-reconstruction/actions/workflows/ci.yml)

A scanning electron microscope builds an image one pixel at a time. If the beam visits only 5–30% of the pixels, acquisition is faster and the specimen receives less dose. The missing pixels then have to be reconstructed.

This repo simulates that acquisition and compares classical and learned reconstructions on real SEM images.

What sets it apart from a generic inpainting benchmark:

- **Physically motivated measurements.** Shot noise is Poisson and scales with dwell time. Sampling patterns respect what scan coils can do: the beam cannot jump anywhere instantly, so full and partial raster lines are compared against the usual uniform-random pixels. A scan-time model charges for every settle and flyback, so methods are compared at equal *acquisition time* as well as at equal pixel count.
- **Classical solvers written from scratch.** Biharmonic interpolation uses its own sparse solve. TV inpainting has two solvers, Chambolle–Pock primal-dual and ADMM, and two data terms, least squares and the Poisson likelihood. All are checked against scikit-image.
- **Learned models.** A U-Net conditioned on the sampling mask is trained across fractions, patterns and doses. A conditional diffusion model is adapted from [diffusion-sparse-reconstruction-hpc](https://github.com/CameronGordonn/diffusion-sparse-reconstruction-hpc).

## Data

The dataset is **NFFA-Europe "100% SEM"** (Aversa et al., CNR-IOM): 21,169 SEM images at 1024×768 in 10 categories (particles, MEMS, nanowires, fibres, …).
- Licence: **CC-BY**. DOI [10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba](https://doi.org/10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba).
- Paper: R. Aversa, M. H. Modarres, S. Cozzini, R. Ciancio, A. Chiusole, *The first annotated set of scanning electron microscopy images for nanoscience*, Sci. Data 5, 180172 (2018).

Preprocessing (`semrecon/data.py`):
- The roughly 230 larger images (2048 or 3072 wide) are resized to 1024 wide so the pixel scale is comparable.
- Most images carry a Zeiss info banner starting at rows 580–690, and about 2% have none. The banner is detected per image as the first nearly all-white row at or below row 570, which avoids false positives from white specimen backgrounds. Only the rows above it are kept.
- Images are converted to grayscale in [0, 1].
- The split is 80/10/10 by image, stratified by category.
- Evaluation uses a frozen set of 512² test crops, one per image, drawn round-robin over categories.
- Masks and noise are never stored. They are regenerated from a seed per (image, pattern, fraction), so every method sees bit-identical inputs.

## Forward model

The clean image x ∈ [0, 1] is treated as the mean detected secondary-electron yield. A measured pixel with dwell time τ, at a beam giving Φ detected electrons per second at x = 1, records

$$n_i \sim \mathrm{Poisson}(D\,x_i),\qquad D = \Phi\tau,\qquad y_i = n_i / D ,$$

so E[y] = x and Var[y] = x/D. Unmeasured pixels are zero and flagged by the mask M.

There are two dose regimes (`semrecon/forward.py`):

| regime | per-pixel dose | question it answers |
|---|---|---|
| `fixed_dwell` (D = 20) | same D at every fraction | how much faster can we scan at the same pixel quality? |
| `fixed_dose` (D_full = 2) | D = D_full / f | at the same total specimen dose, is it better to measure fewer pixels more carefully? |

## Sampling patterns and scan time

![patterns](results/figures/patterns.png)

The fast-scan direction runs along rows (`semrecon/patterns.py`). Each generator hits the requested fraction exactly:

- **uniform**: iid pixels. This is the standard inpainting setup. Physically, almost every pixel needs a blanked jump followed by coil settling.
- **partial raster**: a subset of full scan lines, one at a random offset within each of ⌈fH⌉ strata. There are no in-line jumps.
- **line-hop**: every line is scanned, but the beam dwells only on random-length segments and hops *forward* over the gaps. The coils never reverse mid-line. The segment count per line is k/16, and the lengths come from a uniform random composition.

The scan-time model is `t = N_pixels·τ + N_jumps·t_settle + N_lines·t_flyback`, with τ = 1 µs, t_settle = 5 µs and t_flyback = 50 µs.

This model changes the comparison substantially. At 10% of pixels, a uniform mask costs **0.58×** the time of a full raster, against 0.20× for line-hop and 0.10× for partial raster. At 20% uniform sampling, the time saving is gone entirely (0.99×).

## Methods

| method | where | notes |
|---|---|---|
| Biharmonic | `baselines/biharmonic.py` | Minimises ‖Lx‖² with x = y on M. L is the Neumann 5-point Laplacian. It solves (L²)_UU x_U = −(L²)_UK y_K with sparse LU. It interpolates exactly, so **shot noise passes straight through**. |
| TV-L2 | `baselines/tv.py` | λ·TV(x) + ½‖M(x−y)‖², with x ∈ [0,1] and isotropic TV. Solved with Chambolle–Pock and warm-started from a nearest-neighbour fill. |
| TV-Poisson | `baselines/tv.py` | λ·TV(x) + Σ_M (x − y log x), the Poisson negative log-likelihood. Same PDHG, using the closed-form KL prox x = ½[(v−τ) + √((v−τ)² + 4τy)]. |
| U-Net | `models/unet.py` | 7.8M parameters. Input is [zero-filled y, M]. It predicts a residual on top of a normalized-convolution fill. One model covers every f ∈ [5%, 30%], pattern, dose and regime. Loss is L1 + 0.2·(1−SSIM). |
| U-Net (uniform-only) | same | Ablation: trained on uniform masks only, then tested on all patterns. |
| Diffusion | `models/diffusion.py` | Ported from the ERA5 project (7.9M parameters). Conditioned by concatenating (x_t, y, M), with the time embedding in every block. Masks are drawn per example, and one model covers all fractions and patterns. Two ᾱ off-by-one bugs from the original are fixed. The consistency mode is chosen on val from three: `none` (pure conditional sampling), `repaint` (observations noised to the current step) and `hard` (the original clean-y paste). The Poisson-noisy y makes pasting the observations a real trade-off. The output is the mean of a 4-sample DDIM ensemble with a std map. |

TV's λ is grid-searched per (regime, pattern, fraction) on the **validation** crops only (`results/tv_lambdas.json`), over 0.0015–1.6 in factors of two. The grid originally stopped at 0.4. At fixed dose, TV-Poisson picked 0.4 in 7 of 15 cases, so the grid was extended. Only line-hop at 30% then moved (to 0.8), and those rows were re-evaluated. No other choice lies on the grid edge.

### TV solvers: PDHG vs ADMM

![convergence](results/figures/tv_convergence.png)

This is TV-L2 on a 256² crop with 10% line-hop sampling. Suboptimality is measured against a 20k-iteration reference.

- **Per iteration**, ADMM converges far faster than PDHG.
- **Per wall-clock second**, the ranking changes. The x-update needs an inner CG solve of (M + ρ∇ᵀ∇)x = b, because the mask stops any fast transform from diagonalising it.
- **The DCT preconditioner** replaces M by its mean f, which makes the system exactly diagonal in the DCT-II basis. It helps only a little per iteration and costs about twice the time per iteration of plain CG. With structured masks, f·I is a poor stand-in for M.
- **In practice**, plain-CG ADMM and PDHG are both adequate. PDHG, warm-started, settles in about 100 iterations; from a zero start it takes more than 1000, because TV only spreads information into a gap about one pixel per iteration.

## Results

**Setup.**
- **Test set:** every method sees the same 100 frozen 512² test crops.
- **Cases:** 3 patterns × 5 fractions × 2 dose regimes, so 30 cases per image and 3,000 reconstructions per method. Masks and noise are identical across methods.
- **Training:** on one A100.
  - U-Net: 60k steps.
  - Uniform-only ablation: 30k steps.
  - Diffusion: 150k steps.
- **Diffusion sampler:** chosen on val. Pure conditional sampling (`none`) scored 23.08 dB, against 18.08 for `repaint` and 17.78 for `hard`.
- **Classical methods:** run on CPU.

Raw rows are in `results/metrics*.csv`.

### Mean over all fractions

PSNR in dB by sampling pattern; SSIM is averaged over all cases.

| method | fixed dwell: uniform | partial raster | line-hop | SSIM | fixed dose: uniform | partial raster | line-hop | SSIM |
|---|---|---|---|---|---|---|---|---|
| Biharmonic | 18.03 | 19.66 | 16.75 | 0.27 | 16.87 | 18.41 | 15.65 | 0.24 |
| TV-L2 | 24.05 | 23.53 | 23.21 | 0.48 | 23.72 | 23.15 | 22.84 | 0.47 |
| TV-Poisson | 23.70 | 23.25 | 22.90 | 0.46 | 23.26 | 22.74 | 22.55 | 0.44 |
| U-Net (uniform-only) | 25.90 | 22.79 | 23.53 | 0.51 | 25.49 | 21.99 | 22.84 | 0.48 |
| **U-Net** | **26.11** | **25.48** | **25.22** | **0.58** | **25.71** | **25.09** | **24.88** | **0.57** |
| Diffusion | 25.79 | 25.08 | 24.83 | 0.55 | 25.36 | 24.64 | 24.46 | 0.54 |

### At 10% sampling

At f = 10% the two regimes coincide, since D = 2 / 0.1 = 20. The ± values are 95% bootstrap half-widths over images. They are mostly image-to-image variation; the paired differences below are much tighter. `results/summary.md` has the SSIM version.

| method | uniform | partial raster | line-hop |
|---|---|---|---|
| Biharmonic | 17.91 ± 0.46 | 19.96 ± 0.61 | 16.27 ± 0.48 |
| TV-L2 | 23.51 ± 1.02 | 22.77 ± 1.02 | 22.47 ± 0.99 |
| TV-Poisson | 23.16 ± 0.92 | 22.59 ± 1.02 | 22.24 ± 1.01 |
| U-Net (uniform-only) | 25.45 ± 1.15 | 21.59 ± 0.88 | 22.40 ± 1.02 |
| **U-Net** | **25.67 ± 1.16** | **24.83 ± 1.20** | **24.61 ± 1.21** |
| Diffusion | 25.30 ± 1.17 | 24.41 ± 1.20 | 24.18 ± 1.20 |

### Paired differences

Each image's PSNR difference is averaged over its cases. The interval is a 95% bootstrap CI over the 100 images.

| comparison | fixed dwell | fixed dose | first method better on |
|---|---|---|---|
| U-Net − TV-L2 | +2.01 [1.78, 2.25] | +1.99 [1.76, 2.24] | 99% of images |
| U-Net − Diffusion | +0.37 [0.33, 0.42] | +0.41 [0.36, 0.45] | 98–99% |
| Diffusion − TV-L2 | +1.64 [1.42, 1.87] | +1.58 [1.37, 1.83] | 98% |
| TV-L2 − TV-Poisson | +0.31 [0.18, 0.45] | +0.39 [0.26, 0.53] | 71–83% |
| U-Net − uniform-only, partial raster | +2.69 [2.26, 3.18] | +3.10 [2.61, 3.63] | 100% |
| U-Net − uniform-only, line-hop | +1.69 [1.40, 2.02] | +2.04 [1.66, 2.46] | 100% |

![psnr vs fraction, fixed dwell](results/figures/psnr_vs_frac_fixed_dwell.png)
![psnr vs fraction, fixed dose](results/figures/psnr_vs_frac_fixed_dose.png)
![qualitative, line-hop 10%](results/figures/qualitative_line_hop_10pct_fixed_dwell.png)

### Equal scan time

A microscopist cares less about the pixel fraction than about acquisition time. For each method, this is the best PSNR reachable within a scan-time budget, taking the best of the 15 (pattern, fraction) configurations that fit.

![best PSNR within a scan-time budget, fixed dwell](results/figures/psnr_frontier_fixed_dwell.png)

| method (fixed dwell) | ≤ 0.1× | ≤ 0.2× | ≤ 0.3× | ≤ 1× |
|---|---|---|---|---|
| Biharmonic | 20.11 (raster 5%) | 20.11 (raster 5%) | 20.11 (raster 5%) | 20.11 (raster 5%) |
| TV-L2 | 22.77 (raster 10%) | 24.27 (raster 20%) | 25.21 (raster 30%) | 25.21 (raster 30%) |
| TV-Poisson | 22.59 (raster 10%) | 23.97 (raster 20%) | 24.64 (raster 30%) | 24.64 (raster 30%) |
| U-Net (uniform-only) | 21.59 (raster 10%) | 23.61 (raster 20%) | 25.47 (raster 30%) | 26.48 (uniform 20%) |
| **U-Net** | **24.83 (raster 10%)** | **26.32 (raster 20%)** | **27.08 (raster 30%)** | **27.08 (raster 30%)** |
| Diffusion | 24.41 (raster 10%) | 26.02 (raster 20%) | 26.85 (raster 30%) | 26.85 (raster 30%) |

- **Partial raster is the right choice at every budget up to a full raster, for every method except the uniform-only ablation.** Biharmonic stays at 5%, because more lines only add noise it can't remove.
- **Past 0.3×, extra time buys almost nothing.** Up to a full raster's time, nothing beats raster at 30%. Only uniform at 30% edges past it, by 0.2 dB, and it takes 1.31×: longer than scanning every pixel.
- **The uniform-only U-Net is the exception:** from about 0.8× it is best with uniform masks, the only kind it was trained on.
- **Fixed dose:** partial raster is again the frontier for the U-Net, diffusion and TV-L2. TV-Poisson and the ablation switch to uniform at 5% once the budget reaches 0.34×. `results/summary.md` has the fixed-dose table and each configuration's exact time. `psnr_vs_time_<regime>.png` shows every pattern's curve per method.

### By specimen category

![PSNR by category, fixed dwell](results/figures/psnr_by_category_fixed_dwell.png)

Each category has 10 test images. The ordering holds in all 10 categories and both regimes: U-Net, then diffusion, then the best classical method.
- **Hardest categories:** fine, dense textures such as porous sponge and coated films (U-Net about 21 dB).
- **Easiest categories:** isolated objects on smooth backgrounds, such as tips and particles (about 30 dB).
- **Where the U-Net gains most over TV-L2:** on fibres, +3.2 dB [2.5, 3.9]; the gain is smallest on biological specimens, +1.2 dB [0.9, 1.5]. The table with paired CIs is in `results/summary.md`.

### Findings

- **The U-Net is best in every one of the 30 cases, in both regimes.**
  - It gains about 2 dB over the best classical method, TV-L2, and is better on 99% of individual images.
  - It is also by far the fastest: 0.03 s per 512² crop on an A100, against about 12 s for TV in one CPU worker process.
- **Diffusion comes second, but it doesn't beat the direct regressor on these metrics.**
  - It trails the U-Net by 0.37–0.41 dB and is worse on 98–99% of images.
  - It takes 4.8 s per crop: 4 samples × 100 DDIM steps.
  - PSNR and SSIM reward the conditional mean, which is exactly what the U-Net is trained to predict. Even a 4-sample ensemble mean only approaches it.
  - Enforcing the measurements during sampling hurts badly, by about 5 dB on val. The observations carry Poisson noise, so pasting them back in, clean or re-noised, injects that noise.
- **Training on scan-feasible patterns matters.**
  - The uniform-only ablation nearly matches the U-Net on uniform masks, trailing by only 0.2 dB.
  - It loses 1.7–3.1 dB on partial raster and line-hop, and falls below TV-L2 on partial raster.
  - In the qualitative panel it invents streak texture along the line-hop segments.
- **At equal scan time, contiguous sampling wins by a wide margin.** Scan time here is relative to a full raster, from the settle and flyback model.
  - In the fixed-dwell regime, partial raster at 30% (0.30× the raster time) beats uniform at 5% (0.34×) for every method: U-Net 27.08 vs 24.64 dB, TV-L2 25.21 vs 22.70 dB.
  - Uniform sampling spends most of its time on coil settling.
  - At fixed dose the margin shrinks, to +0.6 dB for the U-Net and +0.7 dB for diffusion. It reverses for biharmonic, TV-Poisson and the uniform-only U-Net, because those 30% of pixels carry only 6.7 counts each.
- **At a fixed total dose, only methods that denoise benefit from spreading it over more pixels.**
  - As f goes from 5% to 30%, the U-Net improves from 24.25 to 25.82 dB and TV-L2 from 22.78 to 23.83 dB.
  - Biharmonic falls from 19.07 to 14.59 dB, because it interpolates the noise exactly.
  - The learned models' gains flatten beyond about 15%.
- **The Poisson data term doesn't help.**
  - TV-Poisson trails TV-L2 by 0.3–0.4 dB, including at fixed dose, where measured pixels get as few as 6.7 counts.
  - The λ grid was extended to rule out a tuning artifact.
  - Possible reasons: at these counts the Gaussian approximation is already adequate, and the reference images are themselves noisy JPEGs.

`python scripts/make_figures.py` writes every figure for both regimes: PSNR, SSIM and unmeasured-pixel PSNR against fraction (`*_vs_frac_<regime>.png`), PSNR against scan time per method (`psnr_vs_time_<regime>.png`), the scan-time budget frontier (`psnr_frontier_<regime>.png`) and the category breakdown (`psnr_by_category_<regime>.png`). It also writes the tables in `results/summary.md`. When the result CSVs cover different image sets, only the shared images are compared.

## What this simulation ignores

- **Scan-coil dynamics.** Hysteresis, overshoot and settling transients are reduced to one constant settle time per jump. Real line-hop scans would show position errors at the start of each segment, and those errors depend on the jump length.
- **Drift.** Stage and beam drift between lines, and over a long acquisition, are absent. Sparse scans taken quickly suffer *less* drift than a full raster, an advantage this simulation does not credit.
- **Charging and beam damage.** Insulating specimens charge nonuniformly, and the charging depends on scan order and dwell. That changes contrast and can deflect the beam. The fixed-dose regime counts total dose, but it does not model damage or charging.
- **Detector response.** The model omits:
  - SE-yield statistics beyond Poisson, such as the excess noise of the SE cascade and scintillator/PMT gain variance;
  - detector bandwidth and the resulting blur along the scan;
  - the beam's point-spread function;
  - quantisation, and offset/gain drift.
- **Ground truth is not truth.** The "clean" images are themselves noisy, JPEG-compressed acquisitions. Metrics measure agreement with another noisy image, and the learned models partly learn JPEG artifacts.

## Pretrained weights

The three trained models are attached to the [`weights-v1` release](https://github.com/CameronGordonn/sparse-scan-SEM-reconstruction/releases/tag/weights-v1). Each is an inference-only checkpoint of about 31 MB, and the release notes list their sha256 checksums. With them you can skip training and go straight to evaluation or `make_figures.py`:

```bash
mkdir -p checkpoints/{unet,unet_uniform_only,diffusion}
for m in unet unet_uniform_only diffusion; do
  curl -L -o checkpoints/$m/best.pt https://github.com/CameronGordonn/sparse-scan-SEM-reconstruction/releases/download/weights-v1/$m.pt
done
```

## Reproduce

```bash
uv venv && uv pip install -e ".[dev]"      # add --extra-index-url https://download.pytorch.org/whl/cpu for CPU torch
pytest -q                                   # forward model, patterns, solvers vs skimage, U-Net, diffusion

python scripts/prepare_data.py --root data/nffa         # ~13.5 GB download + splits + frozen eval crops
python scripts/evaluate.py                               # tune TV lambda on val, run classical methods (CPU, ~11 h with 5 workers)

# GPU. The optional uint8 cache decodes the training split once (~13 GB) so data loading keeps up with an A100.
python scripts/cache_images.py --out /tmp/train_uint8.npy
python scripts/train_unet.py --config configs/unet.yaml --set data.cache=/tmp/train_uint8.npy
python scripts/train_unet.py --config configs/unet.yaml --out-dir checkpoints/unet_uniform_only \
    --set data.cache=/tmp/train_uint8.npy --set "sampling.patterns=[uniform]" --set train.steps=30000
python scripts/train_diffusion.py --config configs/diffusion.yaml --set data.cache=/tmp/train_uint8.npy

python scripts/evaluate.py --stage eval --methods unet --unet-ckpt checkpoints/unet/best.pt --device cuda --out results/metrics_unet.csv
python scripts/evaluate.py --stage eval --methods unet --unet-ckpt checkpoints/unet_uniform_only/best.pt \
    --method-name unet_uniform_only --device cuda --out results/metrics_unet_uniform_only.csv
python scripts/evaluate.py --stage eval --methods diffusion --diffusion-ckpt checkpoints/diffusion/best.pt \
    --device cuda --out results/metrics_diffusion.csv    # ~4 h on an A100; split with --image-start/--n-images

python scripts/make_figures.py --unet-ckpt checkpoints/unet/best.pt \
    --unet-uniform-ckpt checkpoints/unet_uniform_only/best.pt --diffusion-ckpt checkpoints/diffusion/best.pt
```

- **Colab:** `colab/train.ipynb` keeps the dataset tars, splits, eval sets, checkpoints and results on Drive, and extracts the images to local disk each session. Run it cell by cell, or run its last section to launch `colab/launch_all.sh`, which trains and evaluates every learned model unattended in two parallel lanes on one GPU.
- **SLURM:** `hpc/sbatch_train_*.sh`.

## Layout

```
src/semrecon/  data, forward, patterns, metrics, evaluate, plots, train_unet, train_diffusion
               baselines/{biharmonic,tv}.py   models/{unet,diffusion}.py
scripts/       prepare_data, cache_images, show_patterns, train_*, evaluate, make_figures
configs/       eval.yaml, unet.yaml, diffusion.yaml, *smoke.yaml
results/       metrics*.csv, tv_lambdas.json, summary.md, figures/
colab/ hpc/ tests/
```

## License

The code and the trained model weights are released under the [Apache License 2.0](LICENSE). The NFFA-Europe SEM dataset is © CNR-IOM and licensed [CC-BY](https://doi.org/10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba). It is not redistributed here; `scripts/prepare_data.py` downloads it from B2SHARE.
