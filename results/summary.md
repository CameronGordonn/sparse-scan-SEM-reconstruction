### PSNR at 10% sampling, fixed dose

| method | uniform | partial raster | line-hop |
|---|---|---|---|
| Biharmonic | 17.91 ± 0.46 | 19.96 ± 0.61 | 16.27 ± 0.48 |
| TV-L2 (PDHG) | 23.51 ± 1.02 | 22.77 ± 1.02 | 22.47 ± 0.99 |
| TV-Poisson (PDHG) | 23.16 ± 0.92 | 22.59 ± 1.02 | 22.24 ± 1.01 |
| U-Net | 25.67 ± 1.16 | 24.83 ± 1.20 | 24.61 ± 1.21 |
| U-Net (uniform-only) | 25.45 ± 1.15 | 21.59 ± 0.88 | 22.40 ± 1.02 |
| Diffusion | 25.30 ± 1.17 | 24.41 ± 1.20 | 24.18 ± 1.20 |

### SSIM at 10% sampling, fixed dose

| method | uniform | partial raster | line-hop |
|---|---|---|---|
| Biharmonic | 0.25 ± 0.02 | 0.32 ± 0.02 | 0.24 ± 0.01 |
| TV-L2 (PDHG) | 0.47 ± 0.04 | 0.45 ± 0.04 | 0.45 ± 0.04 |
| TV-Poisson (PDHG) | 0.44 ± 0.03 | 0.45 ± 0.04 | 0.45 ± 0.04 |
| U-Net | 0.58 ± 0.04 | 0.55 ± 0.04 | 0.55 ± 0.04 |
| U-Net (uniform-only) | 0.57 ± 0.04 | 0.38 ± 0.04 | 0.44 ± 0.04 |
| Diffusion | 0.55 ± 0.04 | 0.52 ± 0.04 | 0.51 ± 0.04 |

### Best PSNR within a scan-time budget, fixed dose

Best (pattern, fraction) whose scan time, relative to a full raster, fits the budget (1% slack); each cell also shows that configuration's actual time.

| method | ≤ 0.1× | ≤ 0.2× | ≤ 0.3× | ≤ 0.5× | ≤ 1× |
|---|---|---|---|---|---|
| Biharmonic | 21.19 (raster 5%, 0.05×) | 21.19 (raster 5%, 0.05×) | 21.19 (raster 5%, 0.05×) | 21.19 (raster 5%, 0.05×) | 21.19 (raster 5%, 0.05×) |
| TV-L2 (PDHG) | 22.77 (raster 10%, 0.10×) | 23.40 (raster 20%, 0.20×) | 23.89 (raster 30%, 0.30×) | 23.89 (raster 30%, 0.30×) | 23.89 (raster 30%, 0.30×) |
| TV-Poisson (PDHG) | 22.59 (raster 10%, 0.10×) | 23.16 (raster 20%, 0.20×) | 23.16 (raster 20%, 0.20×) | 23.48 (uniform 5%, 0.34×) | 23.48 (uniform 5%, 0.34×) |
| U-Net | 24.83 (raster 10%, 0.10×) | 25.64 (raster 20%, 0.20×) | 25.86 (raster 30%, 0.30×) | 25.86 (raster 30%, 0.30×) | 25.86 (raster 30%, 0.30×) |
| U-Net (uniform-only) | 21.81 (raster 5%, 0.05×) | 22.40 (line-hop 10%, 0.20×) | 23.02 (raster 30%, 0.30×) | 25.04 (uniform 5%, 0.34×) | 25.64 (uniform 20%, 0.99×) |
| Diffusion | 24.41 (raster 10%, 0.10×) | 25.29 (raster 20%, 0.20×) | 25.55 (raster 30%, 0.30×) | 25.55 (raster 30%, 0.30×) | 25.55 (raster 30%, 0.30×) |

### PSNR by specimen category, fixed dose

Mean over all patterns and fractions. Hardest category first, by U-Net PSNR. The last column is the paired U-Net − TV-L2 gain with a 95% bootstrap CI over images.

| category | n | Biharmonic | TV-L2 (PDHG) | TV-Poisson (PDHG) | U-Net | U-Net (uniform-only) | Diffusion | U-Net − TV-L2 |
|---|---|---|---|---|---|---|---|---|
| Porous Sponge | 10 | 15.60 | 19.20 | 19.09 | 20.58 | 19.65 | 20.13 | +1.38 [+1.13, +1.63] |
| Films Coated Surface | 10 | 15.19 | 19.81 | 19.61 | 21.20 | 19.78 | 20.75 | +1.38 [+1.00, +1.81] |
| Powder | 10 | 16.38 | 20.69 | 20.43 | 22.32 | 21.59 | 21.80 | +1.63 [+1.15, +2.08] |
| Patterned surface | 10 | 16.73 | 21.30 | 21.19 | 23.53 | 22.47 | 23.10 | +2.23 [+1.46, +3.07] |
| MEMS devices and electrodes | 10 | 18.38 | 23.08 | 22.44 | 24.87 | 24.03 | 24.60 | +1.79 [+1.23, +2.42] |
| Nanowires | 10 | 17.22 | 22.76 | 21.80 | 25.02 | 23.81 | 24.62 | +2.26 [+1.50, +3.01] |
| Biological | 10 | 16.46 | 24.58 | 24.57 | 25.71 | 23.02 | 25.38 | +1.13 [+0.89, +1.45] |
| Fibres | 10 | 16.92 | 25.71 | 25.78 | 28.82 | 25.02 | 28.28 | +3.11 [+2.46, +3.81] |
| Particles | 10 | 18.00 | 27.20 | 26.49 | 29.66 | 26.86 | 29.33 | +2.46 [+1.47, +3.53] |
| Tips | 10 | 18.87 | 28.03 | 27.13 | 30.55 | 28.19 | 30.20 | +2.52 [+1.67, +3.46] |

### PSNR at 10% sampling, fixed dwell

| method | uniform | partial raster | line-hop |
|---|---|---|---|
| Biharmonic | 17.91 ± 0.46 | 19.96 ± 0.62 | 16.27 ± 0.48 |
| TV-L2 (PDHG) | 23.51 ± 1.02 | 22.77 ± 1.02 | 22.47 ± 0.99 |
| TV-Poisson (PDHG) | 23.16 ± 0.95 | 22.59 ± 1.02 | 22.24 ± 1.01 |
| U-Net | 25.67 ± 1.16 | 24.83 ± 1.20 | 24.61 ± 1.21 |
| U-Net (uniform-only) | 25.45 ± 1.15 | 21.59 ± 0.88 | 22.40 ± 1.02 |
| Diffusion | 25.30 ± 1.17 | 24.41 ± 1.20 | 24.18 ± 1.20 |

### SSIM at 10% sampling, fixed dwell

| method | uniform | partial raster | line-hop |
|---|---|---|---|
| Biharmonic | 0.25 ± 0.02 | 0.32 ± 0.02 | 0.24 ± 0.01 |
| TV-L2 (PDHG) | 0.47 ± 0.04 | 0.45 ± 0.04 | 0.45 ± 0.04 |
| TV-Poisson (PDHG) | 0.44 ± 0.03 | 0.45 ± 0.04 | 0.45 ± 0.04 |
| U-Net | 0.58 ± 0.04 | 0.55 ± 0.04 | 0.55 ± 0.04 |
| U-Net (uniform-only) | 0.57 ± 0.04 | 0.38 ± 0.04 | 0.44 ± 0.04 |
| Diffusion | 0.55 ± 0.04 | 0.52 ± 0.04 | 0.51 ± 0.04 |

### Best PSNR within a scan-time budget, fixed dwell

Best (pattern, fraction) whose scan time, relative to a full raster, fits the budget (1% slack); each cell also shows that configuration's actual time.

| method | ≤ 0.1× | ≤ 0.2× | ≤ 0.3× | ≤ 0.5× | ≤ 1× |
|---|---|---|---|---|---|
| Biharmonic | 20.11 (raster 5%, 0.05×) | 20.11 (raster 5%, 0.05×) | 20.11 (raster 5%, 0.05×) | 20.11 (raster 5%, 0.05×) | 20.11 (raster 5%, 0.05×) |
| TV-L2 (PDHG) | 22.77 (raster 10%, 0.10×) | 24.27 (raster 20%, 0.20×) | 25.21 (raster 30%, 0.30×) | 25.21 (raster 30%, 0.30×) | 25.21 (raster 30%, 0.30×) |
| TV-Poisson (PDHG) | 22.59 (raster 10%, 0.10×) | 23.97 (raster 20%, 0.20×) | 24.64 (raster 30%, 0.30×) | 24.64 (raster 30%, 0.30×) | 24.64 (raster 30%, 0.30×) |
| U-Net | 24.83 (raster 10%, 0.10×) | 26.32 (raster 20%, 0.20×) | 27.08 (raster 30%, 0.30×) | 27.08 (raster 30%, 0.30×) | 27.08 (raster 30%, 0.30×) |
| U-Net (uniform-only) | 21.59 (raster 10%, 0.10×) | 23.61 (raster 20%, 0.20×) | 25.47 (raster 30%, 0.30×) | 25.90 (line-hop 30%, 0.44×) | 26.48 (uniform 20%, 0.99×) |
| Diffusion | 24.41 (raster 10%, 0.10×) | 26.02 (raster 20%, 0.20×) | 26.85 (raster 30%, 0.30×) | 26.85 (raster 30%, 0.30×) | 26.85 (raster 30%, 0.30×) |

### PSNR by specimen category, fixed dwell

Mean over all patterns and fractions. Hardest category first, by U-Net PSNR. The last column is the paired U-Net − TV-L2 gain with a 95% bootstrap CI over images.

| category | n | Biharmonic | TV-L2 (PDHG) | TV-Poisson (PDHG) | U-Net | U-Net (uniform-only) | Diffusion | U-Net − TV-L2 |
|---|---|---|---|---|---|---|---|---|
| Porous Sponge | 10 | 16.68 | 19.56 | 19.57 | 20.96 | 20.15 | 20.51 | +1.40 [+1.15, +1.65] |
| Films Coated Surface | 10 | 16.30 | 20.13 | 20.04 | 21.63 | 20.40 | 21.22 | +1.49 [+1.11, +1.93] |
| Powder | 10 | 17.58 | 21.09 | 21.03 | 22.78 | 22.17 | 22.28 | +1.69 [+1.18, +2.19] |
| Patterned surface | 10 | 17.91 | 21.64 | 21.63 | 24.01 | 22.99 | 23.80 | +2.38 [+1.66, +3.14] |
| MEMS devices and electrodes | 10 | 19.48 | 23.41 | 23.03 | 25.12 | 24.35 | 24.74 | +1.71 [+1.19, +2.32] |
| Nanowires | 10 | 18.27 | 23.13 | 22.31 | 25.44 | 24.42 | 25.06 | +2.30 [+1.52, +3.04] |
| Biological | 10 | 17.69 | 24.80 | 24.74 | 25.96 | 23.77 | 25.63 | +1.15 [+0.92, +1.46] |
| Fibres | 10 | 18.17 | 26.11 | 26.09 | 29.28 | 26.00 | 28.86 | +3.17 [+2.52, +3.88] |
| Particles | 10 | 19.25 | 27.62 | 26.83 | 30.01 | 27.61 | 29.69 | +2.39 [+1.46, +3.41] |
| Tips | 10 | 20.12 | 28.45 | 27.55 | 30.88 | 28.87 | 30.55 | +2.43 [+1.63, +3.29] |
