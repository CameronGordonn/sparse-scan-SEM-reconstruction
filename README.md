# Sparse-scan SEM reconstruction

[![CI](https://github.com/CameronGordonn/sparse-scan-SEM-reconstruction/actions/workflows/ci.yml/badge.svg)](https://github.com/CameronGordonn/sparse-scan-SEM-reconstruction/actions/workflows/ci.yml)

A scanning electron microscope (SEM) builds an image one pixel at a time, by moving an electron beam across the specimen. Measuring only a fraction of the pixels makes the scan faster and exposes delicate specimens to less of the beam. The skipped pixels then have to be filled in by software.

This project asks two questions. Which software fills in the gaps best? And which pixels should the microscope measure to give it the best chance?

![a fibre image scanned at 20% of its lines, filled in three ways](results/figures/reconstruct_example.png)

*A test image scanned at 20% of its lines, in a fifth of the time of a full scan. The neural network (right) recovers the fibres; the two traditional methods give a noisy or blurred result. The dB score measures closeness to the original; higher is better.*

**New to the topic?** Read the plain-language write-up, [`docs/results-explained.pdf`](docs/results-explained.pdf). It explains the study and every finding with the key figures. The full methods, tables and statistics are in [`docs/technical-details.md`](docs/technical-details.md).

## What we found

1. **A trained neural network (a U-Net) fills in the gaps best in simulation.** It beat the traditional methods in every one of the 30 test conditions, by about 2 dB (over a third less error), and it takes 0.03 seconds per image.
2. **Skip whole lines, not scattered pixels.** For the same scan time, measuring a subset of complete lines gives far better images than measuring random pixels. Random pixels make the beam jump before almost every measurement, which wastes time and is sensitive to imprecise beam landing.
3. **A larger generative model (diffusion) can match the U-Net but not beat it,** and only at about 600 times the cost.
4. **A network has to be trained on the scan patterns it will be used with.** One trained only on random pixels does worse than a traditional method on line scans.
5. **On a real microscope, the network wins only when the real noise resembles its training noise.** On real scans at short exposure times, the noise behaves differently from the simulation's, and a traditional method beats the network. At longer exposures the network wins again.

The main open question is whether training the network on realistic noise closes that gap. The evidence suggests it would, but it hasn't been tested yet; see [Open questions](docs/technical-details.md#open-questions).

## How it works

The study takes 21,169 real SEM images from the public [NFFA-Europe collection](https://doi.org/10.23728/b2share.80df8606fcdb4b2bae1656f0dc6db8ba), simulates a sparse scan of each, and scores how well each method recovers the original. The simulation includes the two things that matter most on a real microscope:
- **Noise.** Each measured pixel counts a random number of electrons, and shorter exposures are noisier.
- **Scan time.** The beam needs time to settle after every jump and to return at the end of every line, so how pixels are chosen changes the time a scan takes.

Three ways of choosing 10% of the pixels, with the time each takes relative to a full scan:

![the three sampling patterns at 10% of pixels](results/figures/patterns.png)

Six methods fill in the gaps:
- three traditional ones: smooth interpolation, and two variants of total-variation denoising;
- two U-Nets, one of them trained only on random pixels as a control;
- a diffusion model.

The findings were then checked on real scans of mouse brain tissue from a public dataset.

## Try it on your own image

You need Python 3.11 or newer. A GPU is optional.

```bash
git clone https://github.com/CameronGordonn/sparse-scan-SEM-reconstruction.git
cd sparse-scan-SEM-reconstruction
pip install -e .
```

Simulate a sparse scan of an image you already have, and compare methods:

```bash
python scripts/reconstruct.py my_image.tif --pattern partial_raster --frac 0.2 --methods biharmonic tv_l2 unet
```

This keeps 20% of the image's lines and fills in the rest three ways. It prints each method's score and saves a side-by-side comparison as `reconstruction.png`. The trained models (about 31 MB each) download automatically on first use. On a CPU it takes under a minute.

If you have a real sparse scan, pass the measured image together with a mask whose non-zero pixels mark what was measured:

```bash
python scripts/reconstruct.py measured.png --mask mask.png --methods unet --save-recon
```

Useful options:
- `--pattern` is one of `uniform`, `partial_raster` or `line_hop`.
- `--frac` is the fraction of pixels to measure.
- `--crop-banner` removes a microscope info bar.
- `python scripts/reconstruct.py --help` lists the rest.

The models were trained on the NFFA images and simulated noise. Expect them to do worse on very different images, or on noise that differs from the simulation's (finding 5).

## Reproduce the results

There are two levels.

**Redraw every results chart from the saved results** (about 10 seconds, no data or GPU needed). All the scores are committed in `results/`:

```bash
python scripts/make_figures.py --skip qualitative convergence
```

This rewrites the charts in `results/figures/` and the tables in `results/summary.md`.

**Rerun the whole study from scratch.** This needs a GPU for the learned models and a day or two of compute in total:

| step | what it does | needs |
|---|---|---|
| 1. Prepare data | downloads the 13.5 GB image set and makes the train/val/test splits | disk space |
| 2. Traditional methods | tunes and scores the three traditional methods | CPU, about 11 h |
| 3. Train the networks | trains the U-Net, the control U-Net and the diffusion model | one A100-class GPU, several hours each |
| 4. Score the networks | scores all three networks on the same test images | GPU, about 4 h for diffusion |
| 5. Follow-up studies | beam-landing errors, the diffusion analysis, the real-microscope check (needs a 9 GB download) | CPU or GPU, about 1 h each |

The exact commands for every step are in [`docs/technical-details.md`](docs/technical-details.md#reproduce). [`colab/train.ipynb`](colab/train.ipynb) runs steps 3 and 4 on Google Colab, and `hpc/` has SLURM scripts. To check the installation, run `pip install -e ".[dev]"` and then `pytest`.

## Repository layout

```
src/semrecon/   the library: data, noise and scan simulation, methods, scoring
scripts/        command-line entry points (reconstruct, evaluate, train, make_figures, ...)
results/        every score as CSV, summary tables, and all figures
docs/           the plain-language write-up and the technical details
configs/ colab/ hpc/ tests/
```

## Credits and license

- **Images:** NFFA-Europe "100% SEM" dataset (Aversa et al., *Sci. Data* 5, 180172, 2018), CC-BY.
- **Real microscope scans:** Chen et al., Zenodo [10.5281/zenodo.20139642](https://doi.org/10.5281/zenodo.20139642), CC-BY-4.0.

Neither dataset is redistributed here; the scripts download them.

The diffusion model is adapted from [diffusion-sparse-reconstruction-hpc](https://github.com/CameronGordonn/diffusion-sparse-reconstruction-hpc).

The code and the trained weights are released under the [Apache License 2.0](LICENSE).
