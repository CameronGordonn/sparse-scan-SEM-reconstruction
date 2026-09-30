120 cases: 20 test images × patterns × fractions.

### Accuracy vs texture

| pattern | PSNR U-Net | PSNR diffusion mean | PSNR one sample | high-freq power U-Net | mean | one sample | LPIPS U-Net | mean | one sample |
|---|---|---|---|---|---|---|---|---|---|
| uniform | 24.45 | 24.50 | 23.22 | 0.56 | 0.41 | 0.61 | 0.417 | 0.535 | 0.452 |
| partial raster | 24.03 | 24.06 | 22.68 | 0.53 | 0.38 | 0.64 | 0.425 | 0.549 | 0.490 |
| line-hop | 23.70 | 23.74 | 22.31 | 0.50 | 0.35 | 0.60 | 0.440 | 0.553 | 0.503 |
| all | 24.06 | 24.10 | 22.74 | 0.53 | 0.38 | 0.62 | 0.427 | 0.546 | 0.482 |

High-frequency power is relative to the ground truth over 0.15–0.5 × Nyquist (1 = same amount of fine detail; above half Nyquist the reference is mostly noise). LPIPS is a learned perceptual distance (lower = closer).

### Ensemble size

| samples averaged | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| PSNR (dB) | 22.72 | 23.39 | 23.78 | 23.99 | 24.10 |

### Uncertainty

| pattern | Spearman(std, abs error) | AUSE (lower is better) | AUSE, random ranking |
|---|---|---|---|
| uniform | 0.28 | 0.0262 | 0.0403 |
| partial raster | 0.28 | 0.0292 | 0.0439 |
| line-hop | 0.30 | 0.0295 | 0.0458 |
| all | 0.29 | 0.0283 | 0.0433 |
