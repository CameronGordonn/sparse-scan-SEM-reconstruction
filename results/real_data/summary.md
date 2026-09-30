### Noise vs dwell time

| dwell (µs) | SNR median [IQR] | noise corr. along rows | across rows |
|---|---|---|---|
| 0.25 | 0.110 [0.104, 0.119] | -0.31 | +0.00 |
| 0.5 | 0.463 [0.436, 0.491] | -0.27 | +0.01 |
| 0.75 | 0.834 [0.791, 0.881] | -0.22 | +0.03 |
| 1 | 2.612 [2.488, 2.742] | -0.03 | +0.10 |
| 1.5 | 4.102 [3.885, 4.346] | +0.03 | +0.14 |
| 2 | 5.579 [5.312, 5.965] | +0.07 | +0.17 |

Power-law fit over all dwells: SNR ∝ dwell^1.96; shot noise alone gives exponent 1. Local exponents between neighbouring dwells: 0.25→0.5 µs: 2.07; 0.5→0.75 µs: 1.45; 0.75→1 µs: 3.97; 1→1.5 µs: 1.11; 1.5→2 µs: 1.07.

### Real partial-raster reconstruction

PSNR against the 2 µs reference after the best affine brightness/contrast map (unsaturated pixels). 'Simulated' replaces the real noise by Poisson noise of the same power.

| method | source | 10% lines | 20% lines | 30% lines |
|---|---|---|---|---|
| Biharmonic | real | 16.05 | 16.06 | 15.67 |
| Biharmonic | simulated | 15.13 | 15.08 | 14.84 |
| TV-L2 (PDHG) | real | 15.77 | 17.03 | 18.30 |
| TV-L2 (PDHG) | simulated | 15.46 | 16.69 | 17.74 |
| U-Net | real | 15.98 | 16.57 | 16.75 |
| U-Net | simulated | 17.05 | 18.45 | 19.03 |
| all lines, raw | real | 14.47 (100% of lines) |  |  |
| all lines, raw | simulated | 14.07 (100% of lines) |  |  |
| all lines + TV-L2 | real | 20.26 (100% of lines) |  |  |
| all lines + TV-L2 | simulated | 18.68 (100% of lines) |  |  |

Equivalent Poisson dose of the real 0.5 µs scan: 22.6 electrons per pixel at mean brightness (20 slices, 512×512 centre crops).


### Equal scan time on real data

Each budget buys all lines at 0.5 µs, half the lines at 1 µs, or a quarter at 2 µs (and so on). PSNR against the mean of the neighbouring 2 µs slices, affine-matched. TV-L2 strength tuned per option on held-out slices. † U-Net outside its training range (> 30% of lines).

| budget | option | TV-L2 (PDHG) | U-Net | Biharmonic | raw |
|---|---|---|---|---|---|
| 0.25× | 0.5 µs, 0.25 of lines | 20.05 | 18.86 | 18.05 | — |
| 0.25× | 1 µs, 0.125 of lines | 20.02 | 20.08 | 20.08 | — |
| 0.25× | 2 µs, 0.0625 of lines | 18.72 | 19.11 | 18.82 | — |
| 0.5× | 0.5 µs, 0.5 of lines | 21.96 | 19.08† | 17.43 | — |
| 0.5× | 1 µs, 0.25 of lines | 21.37 | 20.72 | 20.42 | — |
| 0.5× | 2 µs, 0.125 of lines | 20.92 | 21.44 | 21.20 | — |
| 1× | 0.5 µs, 1 of lines | 22.59 | — | — | 16.93 |
| 1× | 1 µs, 0.5 of lines | 22.10 | 20.92† | 19.87 | — |
| 1× | 2 µs, 0.25 of lines | 22.68 | 23.12 | 22.67 | — |
