
<h1 align="center">
  <strong>History-Enriched Rollout Training<br>for Long-Horizon Autoregressive Neural Operators</strong>
</h1>

<p align="center">
  Jiaquan Zhang · Shuxu Chen · Haifan Meng · Yi Lu · Zhihan Lyu<br>
  Fan Mo · Wei Dong · Yang Yang · Chaoning Zhang
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.29135"><img src="https://img.shields.io/badge/arXiv-2607.29135-B31B1B.svg" height="20" alt="arXiv: 2607.29135"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg" height="20" alt="PyTorch 2.6.0"></a>
  <a href="https://github.com/tum-pbs/apebench"><img src="https://img.shields.io/badge/APEBench-0.1.1-356B8C.svg" height="20" alt="APEBench 0.1.1"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.29135">Paper</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#benchmarks-and-data">Benchmarks</a> ·
  <a href="#training-and-evaluation">Training &amp; Evaluation</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <a href="assets/paper/figure-3.png">
    <img src="assets/paper/figure-3.png" width="680" alt="HERO training: lagged, perturbed-input, and detached current rollouts are ranked to select a fixed failure reference for the relative loss.">
  </a>
</p>
<p align="center">
  <em>HERO learns from lagged, perturbed-input, and current rollouts through a fixed failure reference. Click to enlarge.</em>
</p>

## Overview

Autoregressive neural operators repeatedly feed their predictions back as inputs, allowing local errors to accumulate over long trajectories. **HERO adds a comparison with the model's own failure behavior to ordinary ground-truth rollout supervision.**

For each training sample, HERO considers three detached candidates: a rollout from a periodically refreshed lagged operator, a rollout from a perturbed input, and the current rollout. It ranks them using rollout error, spectral discrepancy, energy error, and error growth, then uses the highest-scoring failure as a fixed reference in a margin-based relative objective.

This repository implements **HERO** on nine periodic PDE benchmarks in 1D, 2D, and 3D. It also includes a separate stage for three benchmarks.

## Highlights

- **History-informed supervision.** A lagged operator supplies earlier failure behavior without storing every training checkpoint.
- **Multiple failure signals.** Candidate selection combines trajectory, spectral, energy, and error-growth diagnostics.
- **A complete experiment workflow.** Generate reference trajectories, train with resumable checkpoints, and evaluate free autoregressive rollouts through the supplied scripts.

## Installation

The reference environment documented for this repository uses **Python 3.13**. Dependencies are pinned in [`requirements.txt`](requirements.txt), including PyTorch 2.6.0, APEBench 0.1.1, JAX 0.5.3 (CPU), and NumPy 2.2.6.

```bash
git clone https://github.com/Alley123456/HERO.git
cd HERO

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the remaining commands from the repository root with this environment active. Reference data generation uses APEBench on the CPU by default. Training supports CPU and NVIDIA CUDA; `--device auto` selects CUDA when available and otherwise CPU. On macOS, use `--device cpu` (the CLI does not expose an MPS option). For a CUDA installation, choose the appropriate **PyTorch 2.6.0** build from the [official installation instructions](https://pytorch.org/get-started/previous-versions/#v260).

## Quick Start

Start with a small **CPU smoke run** to exercise data generation, HERO training, checkpoint selection, and test evaluation:

```bash
python -m data.generate --pde burgers --smoke
python train.py --pde burgers --device cpu --smoke
```

This uses two trajectories per split and four optimizer updates. It writes data to `datasets/hero_smoke/burgers/` and results to `runs/hero_smoke/burgers/seed_0/`, keeping the smoke run separate from full experiments. It is a workflow check, not a benchmark result.

For a full Burgers run:

```bash
python -m data.generate --pde burgers
python train.py --pde burgers --device auto --seed 0
```

Training automatically evaluates the best validation checkpoint on the test split. Results are saved to `runs/hero/burgers/seed_0/`.

## Benchmarks and Data

Datasets are generated locally with [APEBench](https://github.com/tum-pbs/apebench); no dataset download is required. Generate one task, several tasks, or all nine:

```bash
python -m data.generate --pde ks
python -m data.generate --pde burgers kdv
python -m data.generate --pde all
```

| Benchmark | `--pde` | Grid | Domain extent per axis | Time step |
| :--- | :--- | :---: | :---: | :---: |
| Dispersion | `dispersion` | 160 | 1 | 0.001 |
| Burgers | `burgers` | 160 | 1 | 0.1 |
| Korteweg–de Vries | `kdv` | 160 | 2π | 0.01 |
| Kuramoto–Sivashinsky | `ks` | 160 | 60 | 0.1 |
| Anisotropic diffusion | `anisotropic_diffusion` | 64 × 64 | 1 | 0.1 |
| Kolmogorov flow | `kolmogorov_flow` | 64 × 64 | 2π | 0.1 |
| Decaying turbulence / Navier–Stokes | `navier_stokes` | 64 × 64 | 1 | 0.1 |
| Swift–Hohenberg | `swift_hohenberg` | 32 × 32 × 32 | 10π | 0.1 |
| Unbalanced advection | `unbalanced_advection` | 32 × 32 × 32 | 1 | 0.1 |

All tasks use periodic grids and a single scalar field. The default HERO splits contain **50 training trajectories × 51 frames**, **10 validation trajectories × 201 frames**, and **30 test trajectories × 201 frames**. Frame counts include the initial condition. Data seeds are fixed at 0, 1, and 773; `train.py --seed` controls model training.

Each `datasets/hero/<pde>/` directory contains `train.npy`, `val.npy`, `test.npy`, and `metadata.json`. Arrays have shape `(trajectories, frames, *grid)` and dtype `float32`. Dataset generation reuses existing files after checking their metadata and array specifications. Use `--output` to choose a different data root and pass the same root to training with `--data-dir`.

## Training and Evaluation

- **Training:** Five-step rollouts, batch size 20, and 10,000 AdamW updates. The learning rate warms up to 0.001 over 2,000 steps, then follows cosine decay. Relative supervision starts at step 4,000.
- **Evaluation:** Validation GM100 selects the best checkpoint for 200-step autoregressive test rollouts. GM100 averages each trajectory's geometric-mean nRMSE over the first 100 steps; stable step counts the consecutive initial steps with nRMSE ≤ 0.1, capped at 100.
- **Memory:** Use `--microbatch` to reduce GPU memory usage while keeping the effective batch size at 20:

```bash
python -m data.generate --pde swift_hohenberg
python train.py --pde swift_hohenberg --device cuda --microbatch 5
```

Results are saved to `runs/hero/<pde>/seed_<seed>/`, including `config.json`, `last.pt`, `best.pt`, `metrics.json`, and `test_errors.npy`. Rerun the same command to resume an interrupted run; use a separate `--out` directory when changing settings. See [`train.py`](train.py) and [`model/hero.py`](model/hero.py) for the full protocol.

## Paper and Reproduction Notes

The [paper](https://arxiv.org/abs/2607.29135) studies both spectral and attention-based backbones. **This repository contains the FNO implementation.** The commands above run one training seed and are not a claim of exact reproduction of every paper result.

In particular, the dense Fourier weights in [`model/fno.py`](model/fno.py) give the current 2D FNO **102,881 parameters**, while [paper Table 8](assets/paper/table-8.png) reports **30,541**. Keep this difference in mind when comparing accuracy or computational cost.


## Citation

If HERO is useful in your research, please cite the [arXiv preprint](https://arxiv.org/abs/2607.29135):

```bibtex
@article{zhang2026hero,
  title   = {{HERO}: History-Enriched Rollout Training for Long-Horizon Autoregressive Neural Operators},
  author  = {Zhang, Jiaquan and Chen, Shuxu and Meng, Haifan and Lu, Yi and Lyu, Zhihan and Mo, Fan and Dong, Wei and Yang, Yang and Zhang, Chaoning},
  journal = {arXiv preprint arXiv:2607.29135},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.29135},
  url     = {https://arxiv.org/abs/2607.29135}
}
```

## Acknowledgments

This implementation uses [PyTorch](https://pytorch.org/) for neural operator training and [APEBench](https://github.com/tum-pbs/apebench) for reference PDE trajectories.
